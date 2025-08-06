import asyncio
from services.notice_service import NoticeService
from core.config import engine
from sqlalchemy.ext.asyncio import AsyncSession

async def test_notice_service():
    """NoticeService 테스트"""
    try:
        async with engine.begin() as conn:
            session = AsyncSession(conn)
            service = NoticeService()
            
            print("🔍 공지사항 목록 조회 테스트...")
            notices = await service.get_all_notices(session)
            print(f"✅ 성공: {len(notices)}개의 공지사항 조회됨")
            
            if notices:
                print(f"📋 첫 번째 공지사항: {notices[0].title}")
                print(f"📎 첨부파일 수: {len(notices[0].attachments)}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_notice_service())
