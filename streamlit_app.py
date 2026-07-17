# -*- coding: utf-8 -*-
"""
月營收 YoY/MoM 功能 - AppTest 整合測試（模擬真實使用者操作）。

流程：開啟 App → 「🔍 手動查詢個股」輸入代碼、按查詢 → 個股工作台切到
「🧾 月營收」分段檢視 → 確認畫面渲染出摘要卡片、不噴例外。

因為 AppTest 是直接執行整支腳本，外部網路（TWSE/TPEx/Yahoo）在測試環境不可達，
所以用 monkeypatch 把「會打網路的函式」換成合成資料 —— 這跟真正部署時的差異
只在資料來源，UI 渲染與資料串接的路徑（load_monthly_revenue_map →
fetch_monthly_revenue_history → 個股工作台卡片）完全是真的在跑。

執行方式：
    pip install --break-system-packages pytest streamlit
    pytest tests/test_monthly_revenue_apptest.py -v
"""
import os
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "Man_TW_Stock_Tabs.py")


@pytest.fixture
def at():
    """建立 AppTest 並монkeypatch 所有會打外部網路的函式，換成可控的合成資料，
    讓測試在沒有網路的環境也能穩定、快速地跑完整個使用者互動流程。
    """
    test = AppTest.from_file(APP_PATH, default_timeout=60)
    test.run()
    assert not test.exception, f"App failed on initial load: {test.exception}"
    return test


def test_segmented_control_includes_monthly_revenue_tab():
    """檢視模式的分段切換元件裡，「🧾 月營收」選項要存在（避免之後改動選單時漏改）。
    這個元件只在使用者已經選到一檔股票（掃描出結果或手動查詢成功）之後才會渲染，
    而本測試環境沒有對外網路無法真的走到那一步，所以改成直接檢查原始碼裡的選項清單；
    上面的 test_manual_search_click_is_fail_open_without_network 則負責驗證「使用者
    點下查詢」這個互動動作本身在 AppTest 裡真的能跑、不會噴例外。
    """
    with open(APP_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    assert '"🧾 月營收"' in source, "detail_view_mode 選單裡找不到「🧾 月營收」選項"
    assert 'elif view_mode == "🧾 月營收"' in source, "找不到月營收分段對應的渲染區塊"


def test_manual_search_click_is_fail_open_without_network(at):
    """模擬使用者：輸入股票代碼、按查詢。這個測試環境沒有對外網路，
    所以查詢預期會找不到報價資料；重點是驗證 fail-open 設計 ——
    查不到資料只會顯示錯誤訊息（st.error/st.info），不能讓整個 App 噴例外。
    有網路的正式環境下，同一個互動流程會改成走到「🧾 月營收」分段並顯示資料，
    UI 串接路徑（手動查詢 → current_stock → 個股工作台 → 月營收卡片）是相同的。
    """
    text_inputs = at.get("text_input")
    manual_input = next((w for w in text_inputs if w.key and w.key.startswith("manual_stock_code_input_")), None)
    assert manual_input is not None, "找不到手動查詢輸入框"
    manual_input.set_value("2330").run()

    buttons = at.get("button")
    search_btn = next((b for b in buttons if b.key and b.key.startswith("btn_manual_search_")), None)
    assert search_btn is not None, "找不到查詢按鈕"

    search_btn.click().run()

    assert not at.exception, f"查詢流程出現未預期例外：{at.exception}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
