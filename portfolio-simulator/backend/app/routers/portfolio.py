"""GET /api/portfolio：组合概览与集中度检查（FR-01 / FR-02）。"""
from fastapi import APIRouter, Request

from ..engine import analyze
from ..presenters import portfolio_payload

router = APIRouter(prefix="/api", tags=["portfolio"])


@router.get("/portfolio")
def get_portfolio(request: Request) -> dict:
    state, instruments, rules = request.app.state.portfolio
    return portfolio_payload(state, instruments, rules, analyze(state, instruments, rules))
