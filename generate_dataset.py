import csv
import random

tables = [
    ("users", ["id", "name", "email", "created_at"]),
    ("orders", ["order_id", "user_id", "amount", "order_date"]),
    ("products", ["product_id", "name", "price", "category"]),
    ("employees", ["emp_id", "first_name", "last_name", "department", "salary"]),
    ("departments", ["dept_id", "dept_name", "manager_id"]),
    ("reviews", ["review_id", "product_id", "user_id", "rating", "comment"]),
    ("sales", ["sale_id", "product_id", "quantity", "sale_date"]),
    ("customers", ["customer_id", "name", "email", "phone"]),
    ("suppliers", ["supplier_id", "name", "contact"]),
    ("inventory", ["inventory_id", "product_id", "stock", "warehouse"])
]

templates = [
    ("List all {col} from the {table} table.", "SELECT {col} FROM {table};"),
    ("Show all records in {table}.", "SELECT * FROM {table};"),
    ("Get the {col1} and {col2} for all {table}.", "SELECT {col1}, {col2} FROM {table};"),
    ("How many records are in {table}?", "SELECT COUNT(*) FROM {table};"),
    ("Find all {table} where {col} = '{val}'.", "SELECT * FROM {table} WHERE {col} = '{val}';"),
    ("List all {table1} with their {table2}.", "SELECT * FROM {table1} JOIN {table2} ON {table1}.{fk} = {table2}.{pk};"),
    ("Get {col1} from {table1} and {col2} from {table2} for related records.", "SELECT {table1}.{col1}, {table2}.{col2} FROM {table1} INNER JOIN {table2} ON {table1}.{fk} = {table2}.{pk};"),
    ("Find {col} in {table} with the highest {col2}.", "SELECT {col} FROM {table} WHERE {col2} = (SELECT MAX({col2}) FROM {table});"),
    ("Show all {table} where {col} is in the list of {col2} from {table2}.", "SELECT * FROM {table} WHERE {col} IN (SELECT {col2} FROM {table2});"),
    ("Create a table named {table} with columns {cols}.", "CREATE TABLE {table} ({cols_def});"),
    ("Update {col} to '{val}' in {table} where {col2} = '{val2}'.", "UPDATE {table} SET {col} = '{val}' WHERE {col2} = '{val2}';"),
    ("Delete from {table} where {col} = '{val}'.", "DELETE FROM {table} WHERE {col} = '{val}';"),
    ("What is the average {col} in {table}?", "SELECT AVG({col}) FROM {table};"),
    ("Get the total {col} for each {col2} in {table}.", "SELECT {col2}, SUM({col}) FROM {table} GROUP BY {col2};"),
    ("List all {table} ordered by {col} descending.", "SELECT * FROM {table} ORDER BY {col} DESC;"),
    ("List all {table} ordered by {col} ascending.", "SELECT * FROM {table} ORDER BY {col} ASC;"),
    ("List all {table} selecting only {col1} ordered by {col} ascending.", "SELECT {col1} FROM {table} ORDER BY {col} ASC;"),
    ("List all {table} selecting only {col1} ordered by {col} descending.", "SELECT {col1} FROM {table} ORDER BY {col} DESC;"),
    ("Select {col1} from {table} ordered by {col} ascending.", "SELECT {col1} FROM {table} ORDER BY {col} ASC;"),
    ("Select {col1} from {table} ordered by {col} descending.", "SELECT {col1} FROM {table} ORDER BY {col} DESC;"),
    ("Get {col1} from {table} ordered by {col} ascending.", "SELECT {col1} FROM {table} ORDER BY {col} ASC;"),
    ("Get {col1} from {table} ordered by {col} descending.", "SELECT {col1} FROM {table} ORDER BY {col} DESC;"),
    ("Show the first 10 records from {table}.", "SELECT * FROM {table} LIMIT 10;"),
    ("List all unique {col} in {table}.", "SELECT DISTINCT {col} FROM {table};"),
]

def random_value():
    return random.choice(["Alice", "Bob", "2023-01-01", "100", "HR", "Electronics", "5", "admin@example.com", "Active", "Pending"])

def random_create_cols(cols):
    types = ["INT", "VARCHAR(255)", "DATE", "FLOAT"]
    return ", ".join(f"{c} {random.choice(types)}" for c in cols)

def schema_text(*table_defs):
    seen = set()
    schema_parts = []
    for table_name, columns in table_defs:
        if table_name not in seen:
            schema_parts.append(
                f"Schema: A table called '{table_name}' with columns {', '.join(columns)}."
            )
            seen.add(table_name)
    return " ".join(schema_parts)

rows = []
for _ in range(3000):
    template = random.choice(templates)
    question_tmpl, sql_tmpl = template
    table1, cols1 = random.choice(tables)
    table2, cols2 = random.choice(tables)
    col1 = random.choice(cols1)
    col2 = random.choice(cols1)
    col3 = random.choice(cols2)
    val = random_value()
    val2 = random_value()
    fk = "user_id" if "user_id" in cols1 and "id" in cols2 else col1
    pk = "id" if "id" in cols2 else col3
    cols_def = random_create_cols(cols1)
    cols_str = ", ".join(cols1)
    if '{table2}' in question_tmpl or '{table2}' in sql_tmpl:
        # Two-table question
        schema = schema_text((table1, cols1), (table2, cols2))
    else:
        # Single-table question
        schema = schema_text((table1, cols1))
    question = question_tmpl.format(
        table=table1, table1=table1, table2=table2,
        col=col1, col1=col1, col2=col2, col3=col3,
        val=val, val2=val2, fk=fk, pk=pk, cols=cols_str, cols_def=cols_def
    )
    sql = sql_tmpl.format(
        table=table1, table1=table1, table2=table2,
        col=col1, col1=col1, col2=col2, col3=col3,
        val=val, val2=val2, fk=fk, pk=pk, cols=cols_str, cols_def=cols_def
    )
    input_text = f"{schema} Question: {question}"
    rows.append({"input": input_text, "output": sql})
    input_text = f"Question: {question}"
    rows.append({"input": input_text, "output": sql})

with open("sql_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["input", "output"])
    writer.writeheader()
    writer.writerows(rows)