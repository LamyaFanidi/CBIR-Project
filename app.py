import streamlit as st
from face_utils import capture_face, capture_face_live
from user_auth import register_user, login_user, login_with_faceid
import tempfile
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="CBIR Login", layout="centered")


def accueil():
    st.title("Bienvenue dans CBIR App")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Se connecter"):
            st.session_state.page = "login"
            st.rerun()

    with col2:
        if st.button("S'inscrire"):
            st.session_state.page = "register"
            st.rerun()


def register():
    st.title("Créer un compte")

    username = st.text_input("Nom d'utilisateur")
    email = st.text_input("Email")
    password = st.text_input("Mot de passe", type="password")
    file = st.file_uploader("Votre image de visage", type=['jpg', 'png'])

    if st.button("Créer mon compte"):
        if username and email and password and file:
            try:
                descripteur = capture_face(file)
                register_user(username, email, password, descripteur)
                st.success("Compte créé avec succès !")
                st.session_state.page = "login"
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")
        else:
            st.error("Merci de remplir tous les champs.")

    if st.button("Retour"):
        st.session_state.page = "accueil"
        st.rerun()


def login():
    st.title("Connexion")

    email = st.text_input("Email")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        if email and password:
            username = login_user(email, password, None)
            if username:
                st.success("Connexion réussie")
                st.session_state.page = "CBIR"
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Email ou mot de passe incorrect.")

    st.markdown("---")

    st.subheader("Ou connectez-vous avec :")

    st.button("Se connecter avec Google", key="google")
    st.button("Se connecter avec Facebook", key="facebook")

    if st.button("Se connecter avec Face ID", key="faceid"):
        try:
            st.info("Activation de la caméra...")
            descripteur_face = capture_face_live()
            username = login_with_faceid(descripteur_face)
            if username:
                st.success(f"Bienvenue {username} (Face ID détecté)")
                st.session_state.page = "CBIR"
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Visage non reconnu")
        except Exception as e:
            st.error(f"Erreur : {e}")

    if st.button("Retour"):
        st.session_state.page = "accueil"
        st.rerun()


def cbir():
    import numpy as np
    from cbir_features import extraire_descripteur, euclidean, manhattan, chebyshev, canberra

    st.title("Recherche d'images CBIR")
    st.write(f"Bienvenue {st.session_state.username}")

    st.markdown("---")

    image_file = st.file_uploader("Upload votre image de recherche", type=['jpg', 'png'])

    if image_file:
        import tempfile
        import os

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(image_file.read())
        tfile_path = tfile.name

        st.image(image_file, caption="Image Requête", width=300)

        # Sélection Descripteur
        descripteur = st.selectbox("Choisissez un descripteur", ["GLCM", "Haralick", "BiT", "All"])

        # Sélection Distance
        distance = st.selectbox("Choisissez une mesure de distance", ["Euclidean", "Manhattan", "Chebyshev", "Canberra"])

        # Choix du nombre d'images à afficher
        top_k = st.slider("Nombre d'images similaires à afficher", min_value=1, max_value=20, value=10)

        if st.button("Rechercher"):
            message = st.empty()
            message.info("Recherche en cours...")

            vecteur_query = extraire_descripteur(tfile_path, 'All')

            features = np.load('./features/features.npy', allow_pickle=True)
            paths = np.load('./features/paths.npy', allow_pickle=True)

            distances = []
            for feature in features:
                if distance == "Euclidean":
                    dist = euclidean(vecteur_query, feature)
                elif distance == "Manhattan":
                    dist = manhattan(vecteur_query, feature)
                elif distance == "Chebyshev":
                    dist = chebyshev(vecteur_query, feature)
                elif distance == "Canberra":
                    dist = canberra(vecteur_query, feature)
                distances.append(dist)

            idx_sorted = np.argsort(distances)[:top_k]

            message.success("Voici les images les plus similaires :")

            for idx in idx_sorted:
                if os.path.exists(paths[idx]):
                    st.image(paths[idx], caption=f"Distance = {distances[idx]:.4f}", width=300)
                else:
                    st.warning(f"Image non trouvée : {paths[idx]}")



    st.markdown("---")

    if st.button("Déconnexion"):
        st.session_state.page = "accueil"
        st.rerun()



def main():
    if 'page' not in st.session_state:
        st.session_state.page = "accueil"

    if st.session_state.page == "accueil":
        accueil()
    elif st.session_state.page == "register":
        register()
    elif st.session_state.page == "login":
        login()
    elif st.session_state.page == "CBIR":
        cbir()


if __name__ == '__main__':
    main()
