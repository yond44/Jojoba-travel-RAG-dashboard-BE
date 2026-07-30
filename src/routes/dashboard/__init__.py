from fastapi import APIRouter

from src.routes.dashboard.revenue_router import router as revenue_router
from src.routes.dashboard.customer_router import router as customer_router
from src.routes.dashboard.booking_router import router as booking_router
from src.routes.dashboard.campaign_router import router as campaign_router
from src.routes.dashboard.package_router import router as package_router
from src.routes.dashboard.review_router import router as review_router
from src.routes.dashboard.destination_router import router as destination_router
from src.routes.dashboard.partner_router import router as partner_router

api_router = APIRouter()

api_router.include_router(revenue_router)
api_router.include_router(customer_router)
api_router.include_router(booking_router)
api_router.include_router(campaign_router)
api_router.include_router(package_router)
api_router.include_router(review_router)
api_router.include_router(destination_router)
api_router.include_router(partner_router)