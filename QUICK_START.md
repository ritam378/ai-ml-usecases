# Quick Start Guide

Get up and running with the AI Case Studies repository in 5 minutes!

## What Works Right Now

### ✅ Immediately Usable

1. **All Documentation** - Read and learn
2. **Text-to-SQL Code** - Run and explore
3. **Sample Database** - Query and test

## 5-Minute Quick Start

### Step 1: Clone the Repository (if not already)

```bash
git clone <your-repo-url>
cd ai-usecases
```

### Step 2: Explore the Documentation

**Start with these (no setup required):**

```bash
# Read the main overview
cat README.md

# Check what's complete
cat COMPLETION_SUMMARY.md

# Study the ML interview framework
open docs/interview-framework.md

# Or browse all docs
ls docs/
```

**Recommended reading order:**
1. [README.md](README.md) - Overview
2. [GETTING_STARTED.md](GETTING_STARTED.md) - Learning paths
3. [docs/interview-framework.md](docs/interview-framework.md) - Interview approach
4. [docs/ml-system-design-guide.md](docs/ml-system-design-guide.md) - System design
5. [docs/glossary.md](docs/glossary.md) - Terminology reference

### Step 3: Run the Text-to-SQL Code

```bash
# Navigate to Text-to-SQL
cd 08_generative-ai-llms/01_text-to-sql

# Install dependencies
pip install -r requirements.txt

# Verify the database exists
ls -lh data/sample_database.db

# If database doesn't exist, create it
cd data
python3 create_database.py
cd ..
```

### Step 4: Test the Code

**Without API Key (Schema Manager Only):**

```bash
# Test schema manager
python3 << 'EOF'
from src.schema_manager import SchemaManager

# Initialize
schema_mgr = SchemaManager('data/sample_database.db')

# Get all tables
print("Tables:", schema_mgr.get_all_tables())

# Get table info
customers = schema_mgr.get_table_info('customers')
print(f"\nCustomers table has {len(customers.columns)} columns")

# Format schema
schema = schema_mgr.format_schema_for_llm(['customers'], include_samples=True)
print("\nSchema:\n", schema[:500])
EOF
```

**With API Key (Full Text-to-SQL):**

```bash
# Set your API key
export OPENAI_API_KEY='your-key-here'
# OR
export ANTHROPIC_API_KEY='your-key-here'

# Test the full system
python3 << 'EOF'
from src.query_generator import TextToSQLGenerator

# Initialize (uses OpenAI by default)
generator = TextToSQLGenerator(
    database_path='data/sample_database.db',
    provider='openai'  # or 'anthropic'
)

# Generate SQL
question = "Show all customers from New York"
result = generator.generate_sql(question)

print("Question:", question)
print("Generated SQL:", result.sql)
print("Explanation:", result.explanation)

# Execute the query
results = generator.execute_query(result.sql)
print(f"\nFound {results['row_count']} customers")
EOF
```

### Step 5: Explore Test Queries

```bash
# View the test queries
python3 -c "
import json
with open('data/test_queries.json') as f:
    queries = json.load(f)

print(f'Total test queries: {len(queries)}\n')
for q in queries[:3]:
    print(f'Q{q[\"id\"]}: {q[\"question\"]}')
    print(f'SQL: {q[\"sql\"][:80]}...\n')
"
```

## What to Explore Next

### For Interview Prep

1. **Study the framework** - [docs/interview-framework.md](docs/interview-framework.md)
2. **Review system design** - [docs/ml-system-design-guide.md](docs/ml-system-design-guide.md)
3. **Learn terminology** - [docs/glossary.md](docs/glossary.md)
4. **Check resources** - [docs/resources.md](docs/resources.md)

### For Learning Text-to-SQL

1. **Read problem statement** - [08_generative-ai-llms/01_text-to-sql/problem_statement.md](08_generative-ai-llms/01_text-to-sql/problem_statement.md)
2. **Study solution approach** - [08_generative-ai-llms/01_text-to-sql/solution_approach.md](08_generative-ai-llms/01_text-to-sql/solution_approach.md)
3. **Review source code** - [08_generative-ai-llms/01_text-to-sql/src/](08_generative-ai-llms/01_text-to-sql/src/)
4. **Examine the database** - [08_generative-ai-llms/01_text-to-sql/data/](08_generative-ai-llms/01_text-to-sql/data/)

### For Contributing

1. **Read guidelines** - [CONTRIBUTING.md](CONTRIBUTING.md)
2. **Pick a case study** - Choose from 27 structured ones
3. **Use Text-to-SQL as reference** - Follow the same quality standard
4. **Submit PR** - Share your implementation!

## Troubleshooting

### Database Not Found

```bash
cd 08_generative-ai-llms/01_text-to-sql/data
python3 create_database.py
```

### Module Import Errors

```bash
# Make sure you're in the right directory
cd 08_generative-ai-llms/01_text-to-sql

# Install dependencies
pip install -r requirements.txt
```

### API Key Issues

```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set it if not
export OPENAI_API_KEY='your-actual-key'
```

## Repository Structure Quick Reference

```
ai-usecases/
├── README.md                          # Start here
├── GETTING_STARTED.md                  # Learning paths
├── COMPLETION_SUMMARY.md               # What's done
├── QUICK_START.md                      # This file
│
├── docs/                              # All documentation
│   ├── interview-framework.md         # How to interview
│   ├── ml-system-design-guide.md      # System design
│   ├── glossary.md                    # Terminology
│   └── resources.md                   # Learning resources
│
├── 08_generative-ai-llms/             # LLM case studies
│   └── 01_text-to-sql/                # Text-to-SQL (70% complete)
│       ├── src/                       # Working Python code ✅
│       ├── data/                      # Database + test data ✅
│       ├── problem_statement.md       # Full problem doc ✅
│       └── solution_approach.md       # Complete solution ✅
│
└── [01-07]_*/                         # Other 27 case studies
    └── */                             # Structured, ready for impl
```

## What's Working vs What's Planned

### ✅ Works Now (Ready to Use)
- All documentation (70,000+ words)
- Text-to-SQL source code (4 modules, 2,000 lines)
- Sample database with 5 tables
- 20 test queries
- Schema manager
- Query validator
- Query generator
- Prompt templates

### 🚧 In Progress (Text-to-SQL)
- Unit tests
- Jupyter notebooks
- Extended documentation

### 📋 Planned (Other Case Studies)
- 27 case studies have structure, need implementation
- Use Text-to-SQL as quality reference

## Summary

**In 5 minutes you can:**
1. ✅ Read comprehensive ML interview guides
2. ✅ Understand the repository structure
3. ✅ Run Text-to-SQL code (with or without API key)
4. ✅ Query the sample database
5. ✅ See working examples

**Ready to start?** Jump to Step 1 above!

---

**Questions?** Check [COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md) for detailed status.
