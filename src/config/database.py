import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

ENV_SETTINGS = get_settings()

_db: AsyncIOMotorDatabase = None


async def connect_db():
    global _db
    try:
        mongo_url = (
            ENV_SETTINGS.mongo_url_dev
            if ENV_SETTINGS.environment == "dev"
            else ENV_SETTINGS.mongo_url_prod
        )
        client = AsyncIOMotorClient(mongo_url)
        _db = client[ENV_SETTINGS.database_name]

        await client.admin.command("ping")
        logger.info(f"✅ Connected to MongoDB: {ENV_SETTINGS.database_name}")

        return _db

    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {str(e)}")
        raise


async def close_db():
    global _db
    if _db is not None:
        _db.client.close()
        logger.info("👋 Disconnected from MongoDB")


def get_db() -> AsyncIOMotorDatabase:
    global _db
    if _db is None:
        raise RuntimeError("Database not connected. Call connect_db() first.")
    return _db