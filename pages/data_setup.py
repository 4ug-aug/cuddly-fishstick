import streamlit as st
import pandas as pd
# postgresql
import psycopg2

st.set_page_config(
        page_title="Data Setup",
        page_icon="🧊",
)

def get_state():
    return st.session_state

# Define a function to connect to an SQL database
def connect_to_database():

    # Add field to enter database name and host
    db_name_col, table_name_col = st.columns(2)
    database_name = db_name_col.text_input("Enter the database name:")
    global table_name
    table_name = table_name_col.text_input("Enter the table name:")

    host_col, port_col = st.columns(2)
    host = host_col.text_input("Enter the host:")
    port = port_col.text_input("Enter the port:", value="5432")
    
    # add field for username
    username = st.text_input("Enter your username:")

    # add field for password
    password = st.text_input("Enter your password:", type="password")

    # Connect to the database
    if st.button("Connect"):
        with st.spinner("Connecting to the database..."):
            try:
                conn = psycopg2.connect(
                    database=database_name,
                    user=username,
                    password=password,
                    host=host,
                    port=port
                )
                st.success("Connected to the database!")
                return conn
            except Exception as e:
                st.error("Failed to connect to the database.")
                st.error(e)
                st.stop()

    return None

    

# Define a function to upload a CSV file
def upload_csv():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    elif "data" in state:
        return state["data"]
    else:
        return None

# Define the Streamlit app
st.title("Data Setup")

# Get the session state
state = get_state()

# Create a sidebar menu
menu = ["Upload CSV", "Connect to SQL Database"]
choice = st.sidebar.selectbox("Select an option", menu)

# Connect to an SQL database
if choice == "Connect to SQL Database":
    conn = connect_to_database()
    if conn is not None:
        state['conn'] = conn
        state["data"] = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Upload a CSV file
elif choice == "Upload CSV":
    state['data'] = upload_csv()

# Display the selected dataset
if state.get("data") is not None:
    st.write("Selected dataset:")
    st.write(state.data)
else:
    st.warning("No dataset selected.")
