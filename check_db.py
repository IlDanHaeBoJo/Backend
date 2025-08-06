#!/usr/bin/env python3
import asyncio
from sqlalchemy import text
from core.config import engine

async def check_notices_table():
    """notices 테이블의 컬럼 구조를 확인합니다."""
    try:
        async with engine.begin() as conn:
            # notices 테이블의 컬럼 목록 조회
            result = await conn.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'notices'
                ORDER BY ordinal_position
            """))
            
            columns = result.fetchall()
            print("📋 notices 테이블 컬럼 구조:")
            print("=" * 60)
            for col in columns:
                print(f"컬럼명: {col[0]}, 타입: {col[1]}, NULL 허용: {col[2]}, 기본값: {col[3]}")
            
            # priority 컬럼이 있는지 확인
            priority_exists = any(col[0] == 'priority' for col in columns)
            print(f"\n🔍 priority 컬럼 존재 여부: {'✅ 있음' if priority_exists else '❌ 없음'}")
            
            if not priority_exists:
                print("\n⚠️  priority 컬럼이 없습니다. 마이그레이션이 필요합니다.")
                
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(check_notices_table())
