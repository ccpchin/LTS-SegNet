from werkzeug.security import check_password_hash, generate_password_hash

# Simulated database record
db_user = {
    "email": "admin@example.com",
    "password": generate_password_hash("securepassword", method='pbkdf2:sha256')
}

# Simulated login function
def login(email, password):
    # Simulate database filter
    if email == db_user["email"]:
        return check_password_hash(db_user["password"], password)
    return False

# SQL Injection Test Input
test_email = "' OR '1'='1"
test_password = "irrelevant"

# Run Test
success = login(test_email, test_password)
if success:
    print("❌ SQL Injection succeeded! You are vulnerable.")
else:
    print("✅ SQL Injection blocked. You are safe.")
