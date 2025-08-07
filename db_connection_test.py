#!/usr/bin/env python3
import asyncio
from sqlalchemy import text
from core.config import engine

async def test_database_connection():
    """데이터베이스 연결과 notices 테이블 구조를 확인합니다."""
    print("🔍 데이터베이스 연결 및 테이블 구조 확인")
    print("=" * 60)
    
    try:
        async with engine.begin() as conn:
            # 1. 데이터베이스 연결 테스트
            print("1️⃣ 데이터베이스 연결 테스트...")
            result = await conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ PostgreSQL 연결 성공: {version.split(',')[0]}")
            
            # 2. notices 테이블 존재 확인
            print("\n2️⃣ notices 테이블 존재 확인...")
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'notices'
                )
            """))
            table_exists = result.fetchone()[0]
            print(f"✅ notices 테이블 존재: {'예' if table_exists else '아니오'}")
            
            if not table_exists:
                print("❌ notices 테이블이 존재하지 않습니다!")
                return
            
            # 3. notices 테이블 컬럼 구조 확인 (ERD 기준)
            print("\n3️⃣ notices 테이블 컬럼 구조 확인 (ERD 기준)...")
            result = await conn.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'notices'
                ORDER BY ordinal_position
            """))
            
            columns = result.fetchall()
            print("📋 현재 notices 테이블 구조:")
            print("-" * 50)
            for col in columns:
                print(f"  {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
            
            # 4. ERD 기준 컬럼 확인
            print("\n4️⃣ ERD 기준 컬럼 확인...")
            erd_columns = {
                'notice_id': 'INT PRIMARY KEY',
                'title': 'VARCHAR(255)',
                'content': 'TEXT',
                'priority': 'INT',
                'author_id': 'INT',
                'view_count': 'INT',
                'created_at': 'timestamp',
                'updated_at': 'timestamp'
            }
            
            current_columns = {col[0] for col in columns}
            missing_columns = set(erd_columns.keys()) - current_columns
            extra_columns = current_columns - set(erd_columns.keys())
            
            if missing_columns:
                print(f"❌ 누락된 컬럼: {missing_columns}")
            else:
                print("✅ 모든 ERD 컬럼이 존재합니다")
                
            if extra_columns:
                print(f"⚠️  추가된 컬럼: {extra_columns}")
            
            # 5. 외래 키 확인
            print("\n5️⃣ 외래 키 확인...")
            result = await conn.execute(text("""
                SELECT 
                    tc.constraint_name, 
                    tc.table_name, 
                    kcu.column_name, 
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name 
                FROM 
                    information_schema.table_constraints AS tc 
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                      AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                      AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' 
                  AND tc.table_name='notices'
            """))
            
            foreign_keys = result.fetchall()
            if foreign_keys:
                print("📋 외래 키 정보:")
                for fk in foreign_keys:
                    print(f"  {fk[2]} -> {fk[3]}.{fk[4]}")
            else:
                print("⚠️  외래 키가 설정되지 않았습니다")
            
            # 6. 샘플 데이터 확인
            print("\n6️⃣ 샘플 데이터 확인...")
            result = await conn.execute(text("SELECT COUNT(*) FROM notices"))
            count = result.fetchone()[0]
            print(f"📊 공지사항 개수: {count}개")
            
            if count > 0:
                result = await conn.execute(text("SELECT * FROM notices LIMIT 1"))
                sample = result.fetchone()
                print(f"📄 샘플 데이터: {sample}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(test_database_connection())
