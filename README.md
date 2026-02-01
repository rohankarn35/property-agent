# 🏠 AI Property Search Agent

A custom-built AI agent that helps users find properties near schools using natural language. Built from scratch with **no AI frameworks** - pure Python + Ollama + PostGIS.

## 🎯 Features

- **Natural Language Understanding** - Ask questions like "Find properties near Rato Bangala School within 2 miles"
- **Fuzzy School Matching** - Handles typos and partial names
- **PostGIS Spatial Queries** - Real geospatial distance calculations
- **Multi-turn Conversations** - Remembers context across messages
- **Smart Clarification** - Asks for missing information

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PROPERTY SEARCH AI AGENT                             │
│                        (Custom Built - No Frameworks)                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  USER INPUT                                                                   │
│      │                                                                        │
│      ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    🧠 LLM (Qwen2.5:7b via Ollama)                   │     │
│  │                                                                      │     │
│  │   • Understands user intent                                         │     │
│  │   • Extracts parameters (school, radius, area)                      │     │
│  │   • Decides which TOOL to call                                      │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│      │                                                                        │
│      │ Selects one of 4 tools:                                               │
│      │                                                                        │
│      ├──────────────────┬──────────────────┬──────────────────┐              │
│      ▼                  ▼                  ▼                  ▼              │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌─────────────────┐     │
│  │   TOOL 1   │   │   TOOL 2   │   │   TOOL 3   │   │     TOOL 4      │     │
│  │            │   │            │   │            │   │                 │     │
│  │  search_   │   │   list_    │   │    ask_    │   │    geocode_     │     │
│  │ properties │   │  schools   │   │clarify     │   │    location     │     │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘   └────────┬────────┘     │
│        │                │                │                   │               │
│        ▼                ▼                ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    ⚙️ INTERNAL FUNCTIONS                            │     │
│  ├─────────────────────────────────────────────────────────────────────┤     │
│  │                                                                      │     │
│  │  _resolve_school()     →  Fuzzy match school name in DB             │     │
│  │  _search_properties()  →  PostGIS ST_DWithin spatial query          │     │
│  │  _list_schools()       →  SQL SELECT all schools                    │     │
│  │  _geocode_location()   →  SQL query to get lat/lon coordinates      │     │
│  │                                                                      │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│        │                                                                      │
│        ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    🗄️ PostgreSQL + PostGIS                          │     │
│  ├─────────────────────────────────────────────────────────────────────┤     │
│  │                                                                      │     │
│  │   schools table          │    parcels table                         │     │
│  │   ├─ id                  │    ├─ parcel_id                          │     │
│  │   ├─ name                │    ├─ address                            │     │
│  │   └─ geom (GEOGRAPHY)    │    ├─ area_sqft                          │     │
│  │                          │    ├─ property_type                      │     │
│  │                          │    └─ geom (GEOGRAPHY)                   │     │
│  │                                                                      │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│        │                                                                      │
│        ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    📤 FORMATTED RESPONSE                            │     │
│  │                                                                      │     │
│  │   "🏠 Found 5 properties within 2 miles of Rato Bangala School..."  │     │
│  │                                                                      │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tool Flow Diagram

```
                              User: "Find properties near Rato Bangala"
                                              │
                                              ▼
                              ┌───────────────────────────┐
                              │   LLM Analyzes Intent     │
                              │   Missing: radius, area   │
                              └───────────────────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────┐
                              │  TOOL: ask_clarification  │
                              │  "What radius in miles?"  │
                              └───────────────────────────┘
                                              │
                                              ▼
                              User: "2 miles, 1000-3000 sqft"
                                              │
                                              ▼
                              ┌───────────────────────────┐
                              │ TOOL: search_properties   │
                              │ school=Rato Bangala       │
                              │ radius=2, area=1000-3000  │
                              └───────────────────────────┘
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
               ┌─────────────────────┐               ┌─────────────────────┐
               │  _resolve_school()  │               │ _search_properties()│
               │  Fuzzy match name   │               │  PostGIS query      │
               │  → Get coordinates  │       ───────▶│  ST_DWithin()       │
               └─────────────────────┘               └─────────────────────┘
                                                              │
                                                              ▼
                                              ┌───────────────────────────┐
                                              │   Format & Return         │
                                              │   "🏠 Found 5 properties" │
                                              └───────────────────────────┘
```

---

## 🛠️ Tools - Detailed Explanation

### Tool 1: `search_properties`

**What it does:**
- Searches for properties within a specified radius of a school
- Filters by property size (sqft range)
- Returns sorted list by distance

**SQL Query Used:**
```sql
SELECT parcel_id, address, area_sqft, property_type,
       ST_Distance(geom::geography, school_point::geography) / 1609.344 as distance_miles
FROM parcels
WHERE ST_DWithin(geom::geography, school_point::geography, radius_meters)
  AND area_sqft BETWEEN min_area AND max_area
ORDER BY distance_miles
```

**Why it's needed:**
- Core functionality - this is the main search feature
- Uses PostGIS `ST_DWithin()` for efficient spatial queries
- `::geography` ensures distance is calculated in meters (real-world distance on Earth's surface)

**Internal functions called:**
1. `_resolve_school()` → Gets school coordinates
2. `_search_properties()` → Executes PostGIS query

---

### Tool 2: `list_schools`

**What it does:**
- Returns all school names from the database
- Helps users know what schools they can search near

**SQL Query Used:**
```sql
SELECT name FROM schools ORDER BY name
```

**Why it's needed:**
- Users don't know what schools exist in the database
- Provides discovery - "What can I search for?"
- Called when user asks "what schools are available?"

**Internal function called:**
- `_list_schools()` → Simple SQL SELECT

---

### Tool 3: `ask_clarification`

**What it does:**
- Asks the user for missing information
- Returns a question to the user (not a database query)

**No SQL Query** - This is a conversation tool

**Why it's needed:**
- Search requires: school_name + radius + area_range
- If user only says "find properties near school X", we're missing radius and area
- Instead of failing, agent asks: "What radius would you like to search?"
- Creates natural, conversational experience

**When called:**
- Missing `radius` → "What radius should I search within?"
- Missing `area` → "What property size range are you looking for?"
- Missing `school_name` → "Which school do you want to search near?"

---

### Tool 4: `geocode_location`

**What it does:**
- Converts a location name to coordinates (latitude/longitude)
- Uses fuzzy matching to find the closest match

**SQL Query Used:**
```sql
SELECT name, 
       ST_Y(geom::geometry) as lat,    -- Extract latitude
       ST_X(geom::geometry) as lon,    -- Extract longitude
       similarity(name, 'user_input') as match_score
FROM schools 
WHERE similarity(name, 'user_input') > 0.3
ORDER BY match_score DESC 
LIMIT 1
```

**Why it's needed:**
- Geocoding converts place names → coordinates
- User might ask "where is Rato Bangala School?"
- Returns exact GPS coordinates from database
- Uses `similarity()` function for fuzzy matching (handles typos)

**Internal function called:**
- `_geocode_location()` → SQL with fuzzy match

---

## 🔄 Internal Functions (Backend)

These are NOT called by LLM directly - they're helper functions:

| Function | Purpose | SQL/Logic |
|----------|---------|-----------|
| `_resolve_school(name)` | Find school + get coordinates | Fuzzy match with `similarity()` |
| `_search_properties(lat, lon, radius, area)` | PostGIS spatial search | `ST_DWithin()` query |
| `_list_schools()` | Get all school names | `SELECT name FROM schools` |
| `_geocode_location(name)` | Get lat/lon for location | `ST_X()`, `ST_Y()` extraction |

---

## 🔧 Tools Quick Reference

| Tool | Purpose | When Called |
|------|---------|-------------|
| `search_properties` | Find properties near school | User provides school + radius + area |
| `list_schools` | Show available schools | User asks "what schools?" |
| `ask_clarification` | Get missing info | Missing radius or area |
| `geocode_location` | Get coordinates | User asks "where is X?" |

---

## 📦 Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Qwen2.5:7b via Ollama |
| **Database** | PostgreSQL 15 + PostGIS 3.4 |
| **Language** | Python 3.10+ |
| **Spatial Functions** | ST_DWithin, ST_Distance, similarity() |

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Ollama (for running the LLM locally)

### 1. Install Ollama

#### Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS:
```bash
brew install ollama
```

#### Windows:
Download from [ollama.com/download](https://ollama.com/download)

### 2. Pull the Qwen2.5:7b Model

```bash
# Start Ollama service
ollama serve

# In another terminal, pull the model (4.7GB download)
ollama pull qwen2.5:7b

# Verify model is available
ollama list
```

You should see:
```
NAME           SIZE     
qwen2.5:7b     4.7 GB
```

### 3. Start the Database

```bash
docker-compose up -d
```

Wait a few seconds for PostgreSQL to initialize.

### 4. Install Python Dependencies

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure Environment

```bash
# Copy example config (already configured for local Docker)
cp .env.example .env
```

### 6. Run the Agent

```bash
python ai_agent.py
```

You should see:
```
🏠 AI Property Search Agent (Powered by Qwen2.5:7b)
==================================================
✅ Test data loaded (Kathmandu/Jawalkhel area)
Type 'quit' to exit, 'reset' to clear history

👤 You: 
```

---

## 🔧 Troubleshooting

### Ollama not running
```bash
# Start Ollama service
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

### Database connection failed
```bash
# Check if Docker container is running
docker ps

# Restart database
docker-compose down
docker-compose up -d

# Wait 10 seconds for PostgreSQL to start
```

### Model too slow
- Qwen2.5:7b requires ~8GB RAM
- For faster inference, use GPU if available
- Alternative smaller model: `ollama pull qwen2.5:3b`

---

## 💬 Example Conversations

### Find Properties
```
👤 You: Find properties near Rato Bangala School within 2 miles between 1000-3000 sqft

🤖 Agent: 🏠 Found 5 properties within 2 miles of Rato Bangala School:
  1. House No. 45, Jawalkhel Main Road - 2,200 sqft - 0.09 miles
  2. Sanepa Residence - 1,200 sqft - 0.23 miles
  ...
```

### List Schools
```
👤 You: What schools are available?

🤖 Agent: 📚 Available schools:
  • Rato Bangala School
  • St. Xavier School Jawalkhel
  • Little Angels School
  • Shuvatara School
  • Ullens School
```

### Geocode Location
```
👤 You: Where is Little Angels School?

🤖 Agent: 📍 Little Angels School is located at:
   • Latitude: 27.675
   • Longitude: 85.32
```

---

## 📁 Project Structure

```
property_search_agent/
├── ai_agent.py         # Main agent (run this)
├── init_db.sql         # Database schema
├── docker-compose.yml  # PostgreSQL container
├── requirements.txt    # Python dependencies
├── .env                # Database config
└── README.md           # This file
```

---

## 🔒 Security Features

- **Parameterized SQL** - Prevents SQL injection
- **Fuzzy Match Threshold** - Requires 50%+ match for schools
- **No AI-generated SQL** - Hardcoded query templates for safety

---

## 📍 Test Data (Kathmandu/Jawalkhel)

**Schools:**
- Rato Bangala School
- St. Xavier School Jawalkhel
- Little Angels School
- Shuvatara School
- Ullens School

**Properties:** 10 locations in Lalitpur (800 - 6,000 sqft)

---

## 🙋 Author

Built as a demonstration of custom AI agent development with spatial database integration.

**Key Skills Demonstrated:**
- Custom LLM tool orchestration (no LangChain/frameworks)
- PostGIS spatial queries
- Fuzzy text matching
- Multi-turn conversation handling
