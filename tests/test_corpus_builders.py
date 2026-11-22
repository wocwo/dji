"""页面语料构建辅助函数测试。"""

from __future__ import annotations

from scripts.build_page_corpus import (
    clean_text,
    infer_doc_type,
    infer_lang,
    infer_model,
    infer_version,
)


def test_infer_specific_models():
    assert infer_model("DJI_Mavic_4_Pro_User_Manual_v1.2_zh-cn") == "DJI Mavic 4 Pro"
    assert infer_model("DJI_Mini_4_Pro_DJI_RC_2_Quick_Start_Guide") == "DJI Mini 4 Pro"
    assert infer_model("DJI_Air_3S_User_Manual_v1.0_zh-cn") == "DJI Air 3S"
    assert infer_model("DJI_Matrice_4D_Series_User_Manual") == "DJI Matrice 4D"
    assert infer_model("DJI_RC-N3_User_Guide") == "DJI RC-N3"


def test_infer_version_lang_type():
    assert infer_version("DJI_Mavic_4_Pro_User_Manual_v1.2_zh-cn") == "v1.2"
    assert infer_lang("DJI_Mavic_4_Pro_User_Manual_v1.2_zh-cn") == "zh"
    assert infer_lang("DJI_Air_3S_User_Manual_v1.0_multi") == "multi"
    assert infer_doc_type("DJI_Mavic_4_Pro_User_Manual_v1.2") == "user_manual"
    assert infer_doc_type("DJI_Mini_4_Pro_Safety_Guidelines") == "safety"
    assert infer_doc_type("DJI_Air_3S_Quick_Start_Guide") == "quick_start"
    assert infer_doc_type("DJI_Case_Study") == "case_study"


def test_clean_text_drops_noise():
    raw = "\x00 \n12\n© 2025 大疆创新 版权所有\n智能飞行电池在低温下需要预热。\n...... 12\n"
    cleaned = clean_text(raw)
    assert "预热" in cleaned
    assert "版权所有" not in cleaned
    assert "...... 12" not in cleaned
