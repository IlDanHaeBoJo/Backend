#!/usr/bin/env python3
import asyncio
from sqlalchemy import text
from core.config import engine

async def add_priority_column():
    """notices 테이블에 priority 컬럼을 추가합니다."""
    try:
        async with engine.begin() as conn:
            # priority 컬럼이 이미 존재하는지 확인
            result = await conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'notices' AND column_name = 'priority'
            """))
            
            if result.fetchone():
                print("✅ priority 컬럼이 이미 존재합니다.")
                return
            
            # priority 컬럼 추가
            await conn.execute(text("""
                ALTER TABLE notices 
                ADD COLUMN priority INTEGER DEFAULT 0
            """))
            
            print("✅ priority 컬럼이 성공적으로 추가되었습니다.")
            
            # 컬럼 추가 확인
            result = await conn.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'notices' AND column_name = 'priority'
            """))
            
            column_info = result.fetchone()
            if column_info:
                print(f"📋 추가된 컬럼 정보:")
                print(f"  컬럼명: {column_info[0]}")
                print(f"  타입: {column_info[1]}")
                print(f"  NULL 허용: {column_info[2]}")
                print(f"  기본값: {column_info[3]}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(add_priority_column())
