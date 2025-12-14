# Problem Statement: Text-to-SQL System

## Business Context

### The Challenge

Organizations have vast amounts of data in relational databases, but accessing this data requires SQL expertise. This creates bottlenecks:

- **Business users** wait days for analysts to write queries
- **Analysts** spend 40% of time writing repetitive SQL
- **Executives** can't answer ad-hoc questions independently
- **Customer support** can't quickly look up customer data

**Cost of Current State:**
- Average wait time for SQL query: 2-3 days
- Data analyst salary: $100K/year
- 30% of analyst time on "simple" SQL queries
- Missed business opportunities due to slow insights

### The Opportunity

A Text-to-SQL system could:
- Reduce query wait time from days to seconds
- Free up 30% of analyst time for complex analysis
- Enable self-service analytics for 1000+ employees
- Accelerate decision-making from days to minutes

**Estimated Impact:**
- $500K/year in analyst time savings
- 10x faster time-to-insight
- 100x more employees empowered to query data

### Real-World Examples

**1. E-commerce Company**
- Problem: Business managers can't quickly check inventory levels
- Solution: Slack bot that answers questions like "How many blue t-shirts size M do we have in SF warehouse?"
- Impact: Inventory queries 50x faster, reduced stockouts by 15%

**2. SaaS Company**
- Problem: Support team can't quickly look up customer subscription details
- Solution: Internal tool translating "When does customer X's subscription expire?" to SQL
- Impact: Support ticket resolution 30% faster

**3. Healthcare Provider**
- Problem: Doctors can't quickly query patient history
- Solution: Natural language interface to EHR database
- Impact: 5 minutes saved per patient consultation

## Problem Definition

### Input
- **Natural Language Question**: "What are the top 5 customers by total purchase amount last month?"
- **Database Schema**: Table structures, relationships, column types
- **Context** (optional): User role, database dialect, past queries

### Output
- **SQL Query**: Valid, executable SQL statement
- **Explanation**: Natural language description of what the query does
- **Confidence Score**: Model's confidence in correctness

### Success Criteria

**Must Have:**
1. Generate syntactically valid SQL
2. Execute without errors on database
3. Return correct results for test questions
4. Sub-3 second latency

**Should Have:**
1. Handle complex queries (multi-table joins, aggregations)
2. Provide query explanations
3. Ask clarifying questions for ambiguous inputs
4. Support multiple SQL dialects

**Nice to Have:**
1. Learn from user feedback
2. Suggest relevant visualizations
3. Explain query performance
4. Auto-optimize slow queries

## Requirements

### Functional Requirements

**FR1: SQL Generation**
- Accept natural language question as input
- Generate valid SQL for target database dialect
- Support SELECT queries (read-only)
- Handle:
  - Simple filters (WHERE)
  - Joins (INNER, LEFT, RIGHT)
  - Aggregations (GROUP BY, HAVING)
  - Sorting (ORDER BY)
  - Limiting (LIMIT, TOP)
  - Subqueries
  - Window functions (basic)

**FR2: Schema Understanding**
- Automatically extract database schema
- Identify table relationships (foreign keys)
- Understand column semantics from names
- Cache schema for performance

**FR3: Query Validation**
- Validate SQL syntax before execution
- Check that tables/columns exist
- Prevent destructive operations (DROP, DELETE, UPDATE)
- Enforce row-level security if applicable

**FR4: Error Handling**
- Retry on syntax errors with error message
- Ask for clarification on ambiguous questions
- Provide helpful error messages
- Fall back gracefully

**FR5: Query Explanation**
- Generate natural language explanation of SQL
- Highlight key operations (joins, filters, aggregations)
- Explain expected result structure

### Non-Functional Requirements

**NFR1: Accuracy**
- Execution accuracy: 90%+ (queries run without errors)
- Result accuracy: 85%+ (queries return correct results)
- Target varies by query complexity

**NFR2: Latency**
- P50: < 1 second
- P95: < 3 seconds
- P99: < 5 seconds

**NFR3: Cost**
- < $0.05 per query (including LLM API costs)
- Total cost < $1000/month for 20K queries

**NFR4: Security**
- Read-only database access
- Respect user permissions
- No PII in logs
- SQL injection prevention

**NFR5: Reliability**
- 99.9% uptime
- Graceful degradation if LLM API unavailable
- Query timeout after 30 seconds

**NFR6: Scalability**
- Support 100 concurrent users
- Handle databases with 1000+ tables
- Process 20K queries/month

## Constraints

### Technical Constraints

1. **Database Access**: Read-only replica only
2. **LLM Choice**: OpenAI GPT-4 or Anthropic Claude
3. **Query Timeout**: 30 seconds max execution time
4. **Result Size**: Max 10K rows returned
5. **Context Length**: Max 8K tokens for schema + examples

### Business Constraints

1. **Budget**: $1000/month for LLM API costs
2. **Timeline**: MVP in 4 weeks
3. **Compliance**: HIPAA/GDPR if handling sensitive data
4. **Languages**: English only initially

### Operational Constraints

1. **No Training**: Can't fine-tune models (no labeled data)
2. **Schema Changes**: Must handle schema evolution
3. **Multi-Tenancy**: Support multiple databases
4. **Audit Trail**: Log all queries for compliance

## Data

### Database Schema

For this case study, we use a sample e-commerce database:

**Tables:**
- `customers` (id, name, email, registration_date, city, country)
- `products` (id, name, category, price, stock_quantity)
- `orders` (id, customer_id, order_date, status, total_amount)
- `order_items` (id, order_id, product_id, quantity, unit_price)
- `reviews` (id, product_id, customer_id, rating, comment, review_date)

**Relationships:**
- orders.customer_id → customers.id
- order_items.order_id → orders.id
- order_items.product_id → products.id
- reviews.product_id → products.id
- reviews.customer_id → customers.id

### Test Queries

We'll evaluate on 100 test queries covering:

**Simple (40%):**
- "Show all customers from New York"
- "List products with price > $100"
- "Count total orders"

**Medium (40%):**
- "Top 10 customers by total order value"
- "Average order value by month"
- "Products with no reviews"

**Complex (20%):**
- "Customers who ordered in Q1 but not Q2"
- "Month-over-month revenue growth"
- "Products purchased together (market basket)"

## Evaluation Metrics

### Primary Metrics

**1. Execution Accuracy (EA)**
```
EA = (Queries that execute without errors) / (Total queries)
```
Target: 90%+

**2. Result Accuracy (RA)**
```
RA = (Queries returning correct results) / (Total queries)
```
Target: 85%+
(Requires labeled test set with ground truth)

**3. Exact Match (EM)**
```
EM = (Generated SQL exactly matches reference SQL) / (Total queries)
```
Target: 50%+ (strict metric)

### Secondary Metrics

**4. User Satisfaction**
- Thumbs up/down rating
- Query edit rate (how often users modify SQL)
- Target: 80% thumbs up

**5. Coverage**
- % of user questions attempted vs. declined
- Target: 95% attempted

**6. Latency**
- P50, P95, P99 latency
- Target: P95 < 3s

**7. Cost**
- Cost per query (API tokens)
- Target: < $0.05/query

### Error Analysis

Categorize errors by type:
- Syntax errors (invalid SQL)
- Semantic errors (wrong table/column names)
- Logic errors (query runs but wrong results)
- Ambiguity (multiple valid interpretations)

## Success Criteria

### MVP (4 weeks)
- [x] 85% execution accuracy on simple queries
- [x] Support for single-table queries
- [x] Basic joins (2 tables)
- [x] Simple aggregations
- [x] < 5 second latency

### V1 (8 weeks)
- [ ] 90% execution accuracy overall
- [ ] 85% result accuracy on test set
- [ ] Multi-table joins (3+ tables)
- [ ] Subqueries and CTEs
- [ ] Query explanation
- [ ] < 3 second P95 latency

### V2 (12 weeks)
- [ ] 95% execution accuracy
- [ ] Clarifying questions for ambiguous input
- [ ] Multiple SQL dialects (MySQL, PostgreSQL)
- [ ] User feedback loop
- [ ] Query optimization suggestions

## Out of Scope (for now)

- Write operations (INSERT, UPDATE, DELETE)
- Schema modifications (CREATE, ALTER, DROP)
- Stored procedures and functions
- Multi-database queries (federated queries)
- Real-time data streaming
- Custom visualizations
- Non-English languages

## Key Challenges

### 1. Schema Understanding
- Large schemas (100+ tables) exceed context limits
- Column names are often cryptic (e.g., "cust_nm")
- Relationships not always explicit (no foreign keys)

**Mitigation:**
- Schema pruning (include only relevant tables)
- Add semantic descriptions to tables/columns
- Embed schema documentation

### 2. Ambiguity Resolution
- "Show sales" → Which table? Which time period?
- "Top customers" → By revenue? By order count?

**Mitigation:**
- Ask clarifying questions
- Use context from past queries
- Provide multiple SQL options

### 3. Complex Query Generation
- Multi-table joins with 5+ tables
- Nested subqueries
- Window functions

**Mitigation:**
- Decompose into simpler queries
- Use chain-of-thought prompting
- Provide more examples for complex patterns

### 4. Cost Control
- GPT-4 is expensive ($0.03 per 1K input tokens)
- Large schemas consume many tokens

**Mitigation:**
- Cache schema embeddings
- Use cheaper models for simple queries
- Implement aggressive schema pruning

### 5. Security
- Prevent SQL injection
- Respect row-level permissions
- Avoid exposing sensitive data

**Mitigation:**
- Parameterized queries
- Read-only database access
- Filter schema to remove sensitive tables

## Interview Discussion Points

When discussing this problem in an interview:

1. **Clarify Scope**: "Are we supporting write operations or read-only?"
2. **Understand Users**: "Who are the primary users? Technical or non-technical?"
3. **Define Success**: "How do we measure if generated SQL is correct?"
4. **Discuss Trade-offs**: "Should we optimize for accuracy or latency?"
5. **Consider Scale**: "How many tables? How many queries per day?"
6. **Security**: "What are the data sensitivity and compliance requirements?"
7. **Failure Modes**: "What happens if the LLM generates invalid SQL?"

## Conclusion

Text-to-SQL is a high-impact application of LLMs with clear business value. Success requires balancing accuracy, latency, and cost while handling the complexity of real-world database schemas and ambiguous natural language.

The key challenges are schema understanding, ambiguity resolution, and cost control. A production system needs robust error handling, security measures, and continuous evaluation.
