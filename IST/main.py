import streamlit as st
import network
import regional
import city
import assortment
import ip
from streamlit_option_menu import option_menu
import login


def add_custom_css():
    st.markdown(
        """
        <style>
        .st-emotion-cache-bm2z3a {
            background-image: linear-gradient(to left bottom, #010002, #201d2b, #303753, #375480, #2874ae);
        }
        .st-emotion-cache-13ln4jf {
            padding: 20px 1rem 1rem;
        }
        h1 {
            font-family: "Source Sans Pro", sans-serif;
            font-weight: 700;
            color: rgb(49, 51, 63);
            padding: 6rem 0px 1rem;
            margin: 0px;
            line-height: 1.2;
        }
        .stSelectbox {
            z-index: 1000 !important;
        }
        .stButton>button {
            width: 30%;
            margin: 5px 0;
           background-image: linear-gradient(to right top, #595b5e, #687579, #798f8c, #96a999, #c0c0a4);
            color: white;
            border: none;
            padding: 10px;
            font-size: 16px;
            text-align: left;
        }
        .stButton>button:hover {
            background-color: green;
        }
        .navbar-container {
            display: flex;
            flex-direction: column;
            width: 250px;
            height: 100vh;
            background-color: #0e1117;
            padding: 1rem;
            position: fixed;
            top: 0;
            left: 0;
        }
        .content-container {
            margin-left: 270px;
            padding: 1rem;
        }
        .st-emotion-cache-6qob1r {
    position: relative;
    height: 100%;
    width: 100%;
    overflow: overlay;
    background: black;
}

h1 {
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 700;
    color: rgb(246 247 255);
    padding: 6rem 0px 1rem;
    margin: 0px;
    line-height: 1.2;
}
p, ol, ul, dl {
    margin: 0px 0px 1rem;
    padding: 0px;
    font-size: 1rem;
    font-weight: 400;
    color: white;
}
.st-emotion-cache-j6qv4b p {
    word-break: break-word;
    margin-bottom: 0px;
    color: black;
}

        </style>
        """,
        unsafe_allow_html=True
    )

def home():
    st.title('Home🎢')
    st.write('Welcome to the Internal Store Transfer Data Processing and Analysis App📈')

def handle_navigation():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login.login()
    else:
        add_custom_css()
        
        with st.sidebar:
            selected = option_menu(
                menu_title=None,
                options=["Home", "Internal Store Transfer", "Assortment", "IP"],
                icons=['house', 'box', 'box', 'gear'],
                menu_icon="cast",
                default_index=0,
                orientation="vertical",
                styles={
                    "container": {"padding": "0!important", "background-color": "black"},
                    "icon": {"color": "orange", "font-size": "25px"},
                    "nav-link": {
                        "font-size": "20px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#eee",
                    },
                    "nav-link-selected": {"background-color": "#565458"},
                }
            )

        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        if selected == 'Home':
            home()
        elif selected == 'Internal Store Transfer':
            ist_option = st.selectbox("Select an option", ["Network", "Regional", "City"], key="ist_selectbox")
            if ist_option == 'Network':
                network.main()
            elif ist_option == 'Regional':
                regional.main()
            elif ist_option == 'City':
                city.main()
        elif selected == 'Assortment':
            assortment.main()
        elif selected == 'IP':
            ip.main()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    handle_navigation()
