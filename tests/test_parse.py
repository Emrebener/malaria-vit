from malaria_vit.data import parse_patient_id


def test_extracts_leading_c_number():
    assert parse_patient_id("C100P61ThinF_IMG_20150918_144104_cell_162.png") == "C100"


def test_same_patient_different_cells_match():
    a = parse_patient_id("C39P4thinF_original_IMG_20150622_105253_cell_7.png")
    b = parse_patient_id("C39P4thinF_original_IMG_20150622_105253_cell_88.png")
    assert a == b == "C39"


def test_uninfected_without_p_part():
    # Many uninfected files have no 'P' part; must still resolve to the C-number.
    assert parse_patient_id("C225ThinF_IMG_20151112_113915_cell_166.png") == "C225"


def test_same_patient_across_classes_groups_together():
    # The same patient contributes both parasitised and uninfected cells (different
    # filename shapes); both must collapse to one id so the patient never spans splits.
    parasitised = parse_patient_id("C39P4thinF_original_IMG_20150622_111326_cell_18.png")
    uninfected = parse_patient_id("C39ThinF_IMG_20151005_163251_cell_12.png")
    assert parasitised == uninfected == "C39"


def test_handles_c_number_with_letter_suffix():
    # 'C51AP12...' should resolve to the leading C-number.
    assert parse_patient_id("C51AP12thinF_IMG_20150724_154243_cell_115.png") == "C51"


def test_fallback_when_no_pattern():
    assert parse_patient_id("weirdname_cell_1.png") == "weirdname"
