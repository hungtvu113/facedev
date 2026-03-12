"""
Module quản lý Database - Lưu trữ thông tin người dùng và face embeddings
"""
import sqlite3
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Person:
    """Thông tin một người trong database"""
    id: int
    name: str
    created_at: str
    embeddings_count: int


class DatabaseManager:
    """
    Quản lý SQLite database cho Face Recognition
    Lưu trữ: Người dùng, Face Embeddings, Lịch sử nhận dạng
    """
    
    def __init__(self, db_path: str = "data/faces.db"):
        self.db_path = db_path
        self._ensure_dir()
        self._init_db()
    
    def _ensure_dir(self):
        """Tạo thư mục nếu chưa tồn tại"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_db(self):
        """Khởi tạo database và các bảng"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng người dùng
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                gender TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng face embeddings (mỗi người có thể có nhiều embeddings)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            )
        ''')
        
        # Bảng lịch sử nhận dạng
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                confidence REAL,
                emotion TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"[Database] Initialized at {self.db_path}")
    
    def add_person(self, name: str, gender: str = None) -> int:
        """
        Thêm người mới vào database
        
        Returns:
            ID của người vừa thêm
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO persons (name, gender) VALUES (?, ?)",
                (name, gender)
            )
            conn.commit()
            person_id = cursor.lastrowid
            print(f"[Database] Added person: {name} (ID: {person_id})")
            return person_id
        except sqlite3.IntegrityError:
            # Người đã tồn tại, lấy ID
            cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
            result = cursor.fetchone()
            return result[0] if result else -1
        finally:
            conn.close()
    
    def add_embedding(self, person_id: int, embedding: np.ndarray,
                      image_path: str = None) -> bool:
        """Thêm face embedding cho một người"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Chuyển numpy array thành bytes (đảm bảo float64 để nhất quán)
            embedding_f64 = embedding.astype(np.float64)
            embedding_bytes = embedding_f64.tobytes()

            cursor.execute(
                "INSERT INTO embeddings (person_id, embedding, image_path) VALUES (?, ?, ?)",
                (person_id, embedding_bytes, image_path)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"[Database] Error adding embedding: {e}")
            return False
        finally:
            conn.close()
    
    def get_all_embeddings(self) -> List[Tuple[int, str, str, np.ndarray]]:
        """
        Lấy tất cả embeddings từ database

        Returns:
            List of (person_id, name, gender, embedding)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT e.person_id, p.name, p.gender, e.embedding
            FROM embeddings e
            JOIN persons p ON e.person_id = p.id
        ''')

        results = []
        for row in cursor.fetchall():
            person_id, name, gender, embedding_bytes = row
            embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
            results.append((person_id, name, gender or "Unknown", embedding))

        conn.close()
        return results
    
    def get_person_embeddings(self, person_id: int) -> List[np.ndarray]:
        """Lấy tất cả embeddings của một người"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT embedding FROM embeddings WHERE person_id = ?",
            (person_id,)
        )
        
        embeddings = []
        for row in cursor.fetchall():
            embedding = np.frombuffer(row[0], dtype=np.float64)
            embeddings.append(embedding)
        
        conn.close()
        return embeddings
    
    def get_all_persons(self) -> List[Person]:
        """Lấy danh sách tất cả người trong database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.id, p.name, p.created_at, COUNT(e.id) as emb_count
            FROM persons p
            LEFT JOIN embeddings e ON p.id = e.person_id
            GROUP BY p.id
        ''')
        
        persons = []
        for row in cursor.fetchall():
            persons.append(Person(
                id=row[0],
                name=row[1],
                created_at=row[2],
                embeddings_count=row[3]
            ))
        
        conn.close()
        return persons

    def delete_person(self, person_id: int) -> bool:
        """Xóa một người và tất cả embeddings của họ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM embeddings WHERE person_id = ?", (person_id,))
            cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"[Database] Error deleting person: {e}")
            return False
        finally:
            conn.close()

    def log_recognition(self, person_id: int, confidence: float, emotion: str = None):
        """Ghi log nhận dạng"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO recognition_logs (person_id, confidence, emotion) VALUES (?, ?, ?)",
            (person_id, confidence, emotion)
        )
        conn.commit()
        conn.close()

    def get_person_by_name(self, name: str) -> Optional[Person]:
        """Tìm người theo tên"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT p.id, p.name, p.created_at, COUNT(e.id) as emb_count
            FROM persons p
            LEFT JOIN embeddings e ON p.id = e.person_id
            WHERE p.name = ?
            GROUP BY p.id
        ''', (name,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return Person(id=row[0], name=row[1], created_at=row[2], embeddings_count=row[3])
        return None

    def get_stats(self) -> Dict:
        """Lấy thống kê database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM persons")
        total_persons = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total_embeddings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM recognition_logs")
        total_logs = cursor.fetchone()[0]

        conn.close()

        return {
            "total_persons": total_persons,
            "total_embeddings": total_embeddings,
            "total_recognitions": total_logs
        }

