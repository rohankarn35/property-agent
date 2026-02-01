"""
AI-Powered Property Search Agent using Ollama (Qwen2.5:7b)

This is a fully agentic AI system that uses LLM for:
- Intent understanding and entity extraction
- Tool selection and orchestration  
- Natural conversation with clarification loops
- Response generation
"""

import json
import re
import os
import sys
import time
import psycopg2
import psycopg2.errors
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import ollama

# Configure logging with custom format
class ColoredFormatter(logging.Formatter):
    COLORS = {'DEBUG': '\033[94m', 'INFO': '\033[92m', 'WARNING': '\033[93m', 'ERROR': '\033[91m'}
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.handlers = [handler]
logger.setLevel(logging.INFO)

# Constants
KM_TO_MILES = 0.621371
SQM_TO_SQFT = 10.7639
MILES_TO_METERS = 1609.344

# Tool definitions for the agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_properties",
            "description": "Search for properties near a school. ONLY call this when you have ALL 4 required fields: school_name, radius_miles, area_min_sqft, area_max_sqft. If ANY is missing, use ask_clarification first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "school_name": {
                        "type": "string",
                        "description": "Name of the school to search near"
                    },
                    "radius_miles": {
                        "type": "number",
                        "description": "Search radius in miles"
                    },
                    "area_min_sqft": {
                        "type": "number",
                        "description": "Minimum property area in square feet (REQUIRED)"
                    },
                    "area_max_sqft": {
                        "type": "number",
                        "description": "Maximum property area in square feet (REQUIRED)"
                    }
                },
                "required": ["school_name", "radius_miles", "area_min_sqft", "area_max_sqft"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_schools",
            "description": "List all available schools in the database. Use when user asks what schools are available or when school name is unclear.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": "Ask the user for missing information. Use when radius or school name is missing or ambiguous.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The clarification question to ask the user"
                    },
                    "missing_field": {
                        "type": "string",
                        "enum": ["radius", "school_name", "area"],
                        "description": "What information is missing"
                    }
                },
                "required": ["question", "missing_field"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "geocode_location",
            "description": "Get the coordinates (latitude, longitude) of a location (school or place) from the database. Use this when you need to know where a place is located.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": "Name of the location/school to get coordinates for"
                    }
                },
                "required": ["location_name"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are a Property Search Assistant that helps users find properties near schools.

You have access to these tools:
1. search_properties - Search for properties near a school
2. list_schools - List all available schools in the database  
3. ask_clarification - Ask the user for missing information
4. geocode_location - Get the coordinates (lat/lon) of a location

CRITICAL RULES FOR search_properties:
- You MUST have ALL 4 of these fields before calling search_properties:
  * school_name - which school to search near
  * radius_miles - search radius in miles
  * area_min_sqft - MINIMUM property area in sqft
  * area_max_sqft - MAXIMUM property area in sqft

- NEVER call search_properties if ANY field is missing!
- If missing, use ask_clarification to ask for it ONE at a time in this order:
  1. First: school_name (if missing)
  2. Then: radius_miles (if missing)
  3. Finally: area_min_sqft AND area_max_sqft (if missing)

EXAMPLES:
- User: "find properties" → Missing ALL! Ask: "Which school do you want to search near?"
- User: "properties near Rato Bangala" → Missing radius + area! Ask: "What radius in miles?"
- User gives radius → Still missing area! Ask: "What property size range (min-max sqft)?"
- User: "1000-3000 sqft" → NOW have all 4! Call search_properties

UNIT HANDLING:
- Radius: "miles" or number = use as-is | "km" = multiply by 0.621371
- Area: "sqft" = use as-is | "sqm" = multiply by 10.7639"""


class AIPropertyAgent:
    """AI-powered property search agent using Ollama.
    
    Attributes:
        db_config: Database connection configuration
        model: Ollama model name (default: qwen2.5:7b)
        max_retries: Maximum connection retry attempts
    """
    
    def __init__(self, db_config: Dict[str, Any], model: str = "qwen2.5:7b") -> None:
        self.db_config: Dict[str, Any] = db_config
        self.model: str = model
        self._conn: Optional[psycopg2.extensions.connection] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.max_retries: int = 3
        self.retry_delay: float = 1.0  # seconds
        
    @property
    def conn(self) -> psycopg2.extensions.connection:
        """Get database connection with automatic retry on failure."""
        if self._conn is None or self._conn.closed:
            self._conn = self._connect_with_retry()
        return self._conn
    
    def _connect_with_retry(self) -> psycopg2.extensions.connection:
        """Establish database connection with exponential backoff retry."""
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                conn = psycopg2.connect(**self.db_config)
                conn.autocommit = True
                return conn
            except psycopg2.OperationalError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    print(f"⚠️  DB connection failed (attempt {attempt}/{self.max_retries}). Retrying in {delay:.1f}s...", flush=True)
                    time.sleep(delay)
        
        # All retries failed
        print(f"\n❌ Database connection failed after {self.max_retries} attempts.", flush=True)
        print(f"   Error: {last_error}", flush=True)
        print(f"\n💡 Make sure the database is running:", flush=True)
        print(f"   docker-compose up -d\n", flush=True)
        raise ConnectionError(f"Could not connect to database: {last_error}")
    
    def close(self) -> None:
        """Close database connection safely."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None
            
    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================
    
    def _resolve_school(self, school_name: str) -> Optional[Dict[str, Union[str, float]]]:
        """Find school by name with fuzzy matching.
        
        Args:
            school_name: Name of school to search for
            
        Returns:
            Dict with 'name', 'lat', 'lon' keys if found, None otherwise
        """
        print(f"    ├─ 🏫 CALL: _resolve_school('{school_name}')", flush=True)
        cursor = self.conn.cursor()
        try:
            # Try exact match first
            print(f"    │  ├─ Trying exact match...", flush=True)
            cursor.execute("""
                SELECT name, ST_Y(geom::geometry) as lat, ST_X(geom::geometry) as lon
                FROM schools WHERE name ILIKE %s LIMIT 1
            """, (f'%{school_name}%',))
            
            result = cursor.fetchone()
            if result:
                print(f"    │  └─ ✅ Exact match: {result[0]} at ({result[1]:.4f}, {result[2]:.4f})", flush=True)
                return {"name": result[0], "lat": float(result[1]), "lon": float(result[2])}
            
            # Try fuzzy match
            print(f"    │  ├─ No exact match, trying fuzzy...", flush=True)
            cursor.execute("""
                SELECT name, ST_Y(geom::geometry) as lat, ST_X(geom::geometry) as lon,
                       similarity(name, %s) as sml
                FROM schools WHERE similarity(name, %s) > 0.2
                ORDER BY sml DESC LIMIT 1
            """, (school_name, school_name))
            
            result = cursor.fetchone()
            if result:
                print(f"    │  └─ ✅ Fuzzy match: {result[0]} (confidence: {result[3]:.2f})", flush=True)
                return {"name": result[0], "lat": float(result[1]), "lon": float(result[2]), 
                        "confidence": float(result[3])}
            print(f"    │  └─ ❌ No match found", flush=True)
            return None
        finally:
            cursor.close()
    
    def _search_properties(self, lat: float, lon: float, radius_miles: float,
                          area_min: Optional[float] = None, 
                          area_max: Optional[float] = None) -> List[Dict[str, Any]]:
        """Execute PostGIS spatial query to find properties within radius.
        
        Args:
            lat: Latitude of center point (school location)
            lon: Longitude of center point
            radius_miles: Search radius in miles
            area_min: Minimum property area in sqft (optional)
            area_max: Maximum property area in sqft (optional)
            
        Returns:
            List of property dictionaries with address, area, distance
        """
        print(f"    ├─ 🟢 CALL: _search_properties(lat={lat:.4f}, lon={lon:.4f}, radius={radius_miles:.2f}mi)", flush=True)
        if area_min and area_max:
            print(f"    │     └─ area_filter: {area_min:.0f}-{area_max:.0f} sqft", flush=True)
        
        cursor = self.conn.cursor()
        radius_meters = radius_miles * MILES_TO_METERS
        print(f"    │  ├─ Converting radius: {radius_miles:.2f} miles → {radius_meters:.0f} meters (for PostGIS)", flush=True)
        
        try:
            query = """
                SELECT parcel_id, address, area_sqft, property_type,
                       ST_Distance(geom::geography, 
                           ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                       ) / %s as distance_miles
                FROM parcels
                WHERE ST_DWithin(geom::geography,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
            """
            params = [lon, lat, MILES_TO_METERS, lon, lat, radius_meters]
            
            if area_min is not None and area_max is not None:
                query += " AND area_sqft BETWEEN %s AND %s"
                params.extend([area_min, area_max])
            
            query += " ORDER BY distance_miles"
            print(f"    │  ├─ Executing PostGIS spatial query (ST_DWithin)...", flush=True)
            cursor.execute(query, params)
            
            results = [{"parcel_id": r[0], "address": r[1], "area_sqft": float(r[2]),
                    "property_type": r[3], "distance_miles": round(float(r[4]), 2)}
                   for r in cursor.fetchall()]
            print(f"    │  └─ ✅ Query returned {len(results)} properties", flush=True)
            return results
        finally:
            cursor.close()
    
    def _list_schools(self) -> List[str]:
        """Get all school names from the database.
        
        Returns:
            List of school names sorted alphabetically
        """
        print(f"    ├─ 📚 CALL: _list_schools()", flush=True)
        cursor = self.conn.cursor()
        try:
            print(f"    │  ├─ Querying schools table...", flush=True)
            cursor.execute("SELECT name FROM schools ORDER BY name")
            schools = [r[0] for r in cursor.fetchall()]
            print(f"    │  └─ ✅ Found {len(schools)} schools: {schools}", flush=True)
            return schools
        finally:
            cursor.close()
    
    def _geocode_location(self, location_name: str) -> Optional[Dict[str, Union[str, float]]]:
        """Get coordinates for a location by querying the database.
        
        Args:
            location_name: Name of location to geocode
            
        Returns:
            Dict with 'name', 'lat', 'lon' if found, None otherwise
        """
        print(f"    ├─ 🌍 CALL: _geocode_location('{location_name}')", flush=True)
        cursor = self.conn.cursor()
        try:
            # Query schools table with fuzzy matching
            print(f"    │  ├─ Executing SQL: SELECT with similarity matching...", flush=True)
            cursor.execute("""
                SELECT name, 
                       ST_Y(geom::geometry) as lat, 
                       ST_X(geom::geometry) as lon,
                       similarity(name, %s) as match_score
                FROM schools 
                WHERE similarity(name, %s) > 0.3
                ORDER BY match_score DESC 
                LIMIT 1
            """, (location_name, location_name))
            
            result = cursor.fetchone()
            if result and result[3] >= 0.5:  # Require 50% match
                print(f"    │  └─ ✅ Found: {result[0]} at ({result[1]:.4f}, {result[2]:.4f}) [match: {result[3]:.0%}]", flush=True)
                return {"name": result[0], "lat": round(float(result[1]), 6), "lon": round(float(result[2]), 6)}
            
            print(f"    │  └─ ❌ No match found for '{location_name}'", flush=True)
            return None
        finally:
            cursor.close()
    
    # =========================================================================
    # TOOL EXECUTOR
    # =========================================================================
    
    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool and return result as string.
        
        Args:
            tool_name: Name of the tool to execute
            args: Dictionary of arguments for the tool
            
        Returns:
            Formatted string result from tool execution
        """
        
        if tool_name == "search_properties":
            school = self._resolve_school(args["school_name"])
            if not school:
                return f"❌ Could not find school '{args['school_name']}'. Available schools: {', '.join(self._list_schools())}"
            
            properties = self._search_properties(
                lat=school["lat"], lon=school["lon"],
                radius_miles=args["radius_miles"],
                area_min=args.get("area_min_sqft"),
                area_max=args.get("area_max_sqft")
            )
            
            if not properties:
                result = f"📍 No properties found within {args['radius_miles']} miles of {school['name']}."
            else:
                result = f"🏠 Found {len(properties)} properties within {args['radius_miles']} miles of **{school['name']}**"
                if args.get('area_min_sqft') and args.get('area_max_sqft'):
                    result += f" ({args['area_min_sqft']:,.0f}-{args['area_max_sqft']:,.0f} sq ft)"
                result += ":\n\n"
                for i, p in enumerate(properties, 1):
                    result += f"{i}. {p['address']} - {p['area_sqft']:,.0f} sq ft - {p['distance_miles']} miles away\n"
            
            print(f"    └─ ✅ TOOL RESULT: Found {len(properties)} properties", flush=True)
            return result
        
        elif tool_name == "list_schools":
            schools = self._list_schools()
            print(f"    └─ ✅ TOOL RESULT: Listed {len(schools)} schools", flush=True)
            return f"📚 Available schools:\n" + "\n".join(f"• {s}" for s in schools)
        
        elif tool_name == "ask_clarification":
            print(f"    └─ ❓ CLARIFICATION NEEDED: {args['missing_field']}", flush=True)
            return f"CLARIFICATION_NEEDED: {args['question']}"
        
        elif tool_name == "geocode_location":
            location = self._geocode_location(args["location_name"])
            if location:
                print(f"    └─ ✅ TOOL RESULT: Found coordinates for {location['name']}", flush=True)
                return f"📍 **{location['name']}** is located at:\n   • Latitude: {location['lat']}\n   • Longitude: {location['lon']}"
            else:
                schools = self._list_schools()
                print(f"    └─ ❌ TOOL RESULT: Location not found", flush=True)
                return f"❌ Location '{args['location_name']}' not found in database.\n\n📚 Available locations:\n" + "\n".join(f"• {s}" for s in schools)
        
        return "Unknown tool"
    
    # =========================================================================
    # MAIN AGENT LOOP
    # =========================================================================
    
    def chat(self, user_message: str) -> str:
        """Process user message with AI reasoning and tool calling.
        
        This is the main entry point for the agent. It:
        1. Adds user message to conversation history
        2. Sends context to LLM for reasoning
        3. Executes any tool calls the LLM makes
        4. Returns the final response
        
        Args:
            user_message: The user's natural language input
            
        Returns:
            Agent's response string
        """
        
        # Step 1: Receive input
        print(f"\n{'─'*60}", flush=True)
        print(f"🔄 STEP 1: Received user input", flush=True)
        print(f"   └─ \"{user_message}\"", flush=True)
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Step 2: Build context
        print(f"\n🧠 STEP 2: Building conversation context", flush=True)
        print(f"   └─ History: {len(self.conversation_history)} messages", flush=True)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history
        
        # Step 3: Call LLM
        print(f"\n🤖 STEP 3: Calling Ollama ({self.model})...", flush=True)
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                options={"temperature": 0.1}
            )
            
            assistant_message = response["message"]
            
            # Step 4: Analyze LLM decision
            print(f"\n💭 STEP 4: LLM made a decision", flush=True)
            
            if assistant_message.get("tool_calls"):
                tool_calls = assistant_message["tool_calls"]
                print(f"   └─ Decision: CALL TOOL(S) - {len(tool_calls)} tool(s)", flush=True)
                
                tool_results = []
                for i, tool_call in enumerate(tool_calls, 1):
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    
                    # Step 5: Execute tool
                    print(f"\n🔧 STEP 5.{i}: Executing tool '{tool_name}'", flush=True)
                    print(f"   ├─ Arguments: {json.dumps(tool_args)}", flush=True)
                    
                    result = self._execute_tool(tool_name, tool_args)
                    
                    # Handle clarification specially
                    if result.startswith("CLARIFICATION_NEEDED:"):
                        question = result.replace("CLARIFICATION_NEEDED: ", "")
                        print(f"   └─ Result: Asking for clarification ({tool_args.get('missing_field', 'info')})", flush=True)
                        print(f"{'─'*60}\n", flush=True)
                        self.conversation_history.append({"role": "assistant", "content": question})
                        return question
                    
                    print(f"   └─ Result: Success", flush=True)
                    tool_results.append(result)
                
                # Step 6: Format response
                print(f"\n📤 STEP 6: Formatting final response", flush=True)
                print(f"{'─'*60}\n", flush=True)
                
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": "\n".join(tool_results)
                })
                return "\n".join(tool_results)
            
            else:
                # No tool call, just text response
                content = assistant_message.get("content", "I'm not sure how to help with that.")
                print(f"   └─ Decision: TEXT RESPONSE (no tool needed)", flush=True)
                print(f"\n📤 STEP 5: Returning text response", flush=True)
                print(f"{'─'*60}\n", flush=True)
                
                self.conversation_history.append({"role": "assistant", "content": content})
                return content
                
        except Exception as e:
            print(f"\n❌ ERROR: {e}", flush=True)
            print(f"{'─'*60}\n", flush=True)
            return f"⚠️ AI Error: {e}. Make sure Ollama is running."
    
    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []


# =============================================================================
# TEST & CLI
# =============================================================================

def setup_test_data(agent: AIPropertyAgent):
    """Insert test data for Kathmandu/Jawalkhel area."""
    cursor = agent.conn.cursor()
    try:
        cursor.execute("DELETE FROM parcels; DELETE FROM schools;")
        
        # Schools in Jawalkhel/Lalitpur area, Kathmandu Valley
        cursor.execute("""
            INSERT INTO schools (name, geom) VALUES
            ('Rato Bangala School', ST_SetSRID(ST_MakePoint(85.3120, 27.6680), 4326)::geography),
            ('St. Xavier School Jawalkhel', ST_SetSRID(ST_MakePoint(85.3140, 27.6720), 4326)::geography),
            ('Little Angels School', ST_SetSRID(ST_MakePoint(85.3200, 27.6750), 4326)::geography),
            ('Shuvatara School', ST_SetSRID(ST_MakePoint(85.3080, 27.6650), 4326)::geography),
            ('Ullens School', ST_SetSRID(ST_MakePoint(85.3250, 27.6800), 4326)::geography)
        """)
        
        # Properties around Jawalkhel area
        parcels = [
            ("JWL001", "House No. 45, Jawalkhel Main Road, Lalitpur", 2200, "residential", 85.3130, 27.6690),
            ("JWL002", "Pulchowk Apartment Complex, Lalitpur", 1500, "residential", 85.3180, 27.6710),
            ("JWL003", "Ekantakuna Housing, Lalitpur", 1800, "residential", 85.3100, 27.6620),
            ("JWL004", "Dhobighat Commercial Plaza, Lalitpur", 5500, "commercial", 85.3050, 27.6600),
            ("JWL005", "Kupondole Heights, Lalitpur", 2800, "residential", 85.3160, 27.6730),
            ("JWL006", "Sanepa Residence, Lalitpur", 1200, "residential", 85.3090, 27.6700),
            ("JWL007", "Patan Durbar Square Shop, Lalitpur", 800, "commercial", 85.3250, 27.6750),
            ("JWL008", "Mangalbazar Villa, Lalitpur", 3200, "residential", 85.3280, 27.6780),
            ("JWL009", "Lagankhel Apartment, Lalitpur", 950, "residential", 85.3220, 27.6680),
            ("JWL010", "Satdobato Business Center, Lalitpur", 6000, "commercial", 85.3300, 27.6550),
        ]
        for p in parcels:
            cursor.execute("""
                INSERT INTO parcels (parcel_id, address, area_sqft, property_type, geom)
                VALUES (%s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography)
            """, p)
        
        agent.conn.commit()
        print("✅ Test data loaded (Kathmandu/Jawalkhel area)", flush=True)
    finally:
        cursor.close()


def run_cli():
    """Interactive CLI."""
    from dotenv import load_dotenv
    load_dotenv()
    
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5433)),
        "dbname": os.getenv("DB_NAME", "property_db"),
        "user": os.getenv("DB_USER", "property_agent"),
        "password": os.getenv("DB_PASSWORD", "agent_password")
    }
    
    print("\n🏠 AI Property Search Agent (Powered by Qwen2.5:7b)")
    print("=" * 50)
    
    agent = AIPropertyAgent(db_config, model="qwen2.5:7b")
    
    try:
        setup_test_data(agent)
        print("Type 'quit' to exit, 'reset' to clear history\n")
        
        while True:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'reset':
                agent.reset()
                print("🔄 Conversation reset")
                continue
                
            response = agent.chat(user_input)
            print(f"\n🤖 Agent: {response}")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        agent.close()


if __name__ == "__main__":
    run_cli()
