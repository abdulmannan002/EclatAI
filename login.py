import streamlit as st
import sqlite3
import hashlib

# Connect to SQLite Database (or create it if it doesn't exist)
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

# Create users table if it doesn't exist
def create_users_table():
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, 
            password TEXT
        )
    ''')
    conn.commit()

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to add a new user to the database
def add_user(username, password):
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hash_password(password)))
    conn.commit()

# Function to check login credentials
def login_user(username, password):
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hash_password(password)))
    return c.fetchone()

# User Authentication and Registration
def login():
    st.title("Login")

    # Tabs for switching between login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.success(f"Welcome {username}! Logging you in...")
                st.session_state['logged_in'] = True
                st.session_state['username'] = username

                # Redirect to the app page after login
                st.session_state['page'] = "app"
                
                # Indirect page reload by setting query params
                st.experimental_set_query_params(logged_in="true")

            else:
                st.error("Invalid username or password.")

    with tab2:
        st.subheader("Register")

        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            else:
                try:
                    add_user(new_username, new_password)
                    st.success("Registration successful! Automatically logging you in...")

                    # Automatically log the user in after registration
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = new_username

                    # Redirect to the app page after registration
                    st.session_state['page'] = "app"
                    st.experimental_set_query_params(logged_in="true")

                except sqlite3.IntegrityError:
                    st.error("Username already exists. Please choose another one.")

# Main function to display the login page
def main():
    create_users_table()  # Ensure the users table exists

    # Check if the user is already logged in
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if 'page' not in st.session_state:
        st.session_state['page'] = "login"  # Default page

    # Handle page redirection
    if st.session_state['page'] == "login":
        login()
    elif st.session_state['page'] == "app":
        app_page()  # Call your app page here

def app_page():
    st.title(f"Welcome to the App, {st.session_state['username']}!")
    st.write("You are now logged in.")
    # Add the rest of your app functionality here

if __name__ == "__main__":
    main()
