
import sys
import generate_sql

# Redirect stdout to a file with UTF-8 encoding
with open("supabase_schema.sql", "w", encoding="utf-8") as f:
    sys.stdout = f
    generate_sql.main()
