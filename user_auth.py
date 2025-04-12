import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import json
from db import get_db_connection


def register_user(username, email, password, face_descriptor):
    conn = get_db_connection()
    cursor = conn.cursor()

    password_hash = generate_password_hash(password)
    descriptor_json = json.dumps(face_descriptor.tolist())

    sql = "INSERT INTO users (username, email, password_hash, face_descriptor, auth_provider) VALUES (%s, %s, %s, %s, %s)"
    val = (username, email, password_hash, descriptor_json, 'local')

    cursor.execute(sql, val)
    conn.commit()
    cursor.close()
    conn.close()


def login_user(email, password, captured_descriptor):
    conn = get_db_connection()
    cursor = conn.cursor()

    sql = "SELECT username, password_hash, face_descriptor FROM users WHERE email=%s"
    cursor.execute(sql, (email,))

    result = cursor.fetchone()
    cursor.fetchall()
    cursor.close()
    conn.close()

    if result:
        username, stored_password, stored_descriptor_json = result
        stored_descriptor = np.array(json.loads(stored_descriptor_json))

        if check_password_hash(stored_password, password):
            if captured_descriptor is None:
                return username  # Login classique : return username
            else:
                from numpy.linalg import norm
                distance = norm(stored_descriptor - captured_descriptor)
                if distance < 0.5:
                    return username
    return None


def login_with_faceid(captured_descriptor):
    conn = get_db_connection()
    cursor = conn.cursor()

    sql = "SELECT username, face_descriptor FROM users"
    cursor.execute(sql)

    results = cursor.fetchall()
    cursor.close()
    conn.close()

    from numpy.linalg import norm

    for username, descriptor_json in results:
        descriptor_db = np.array(json.loads(descriptor_json))
        distance = norm(descriptor_db - captured_descriptor)
        if distance < 0.5:
            return username
    return None
