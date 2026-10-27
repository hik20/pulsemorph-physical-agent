"""Local tool description; model proposals cannot change this vocabulary."""
from copy import deepcopy

_TOOL = {
    'id': 'g1_wave_high5_phrase_v1', 'label': 'Wave → High5 · 8 saniye',
    'experimental': True, 'execution_s': 8, 'reference_horizon_s': 8.14,
    'supported_order': ['wave', 'high5'], 'transition_s': [3.5, 4.5],
    'heading_policies': ['recorded_world', 'align_join_heading'],
    'gain_fields': ['first_left_gain', 'first_right_gain', 'second_left_gain', 'second_right_gain'],
    'gain_bounds': [.75, 1.25], 'feature_windows_s': [[1.2, 3.2], [5.2, 7.2]],
    'feature_tolerances': [.08] * 4,
    'physical_screen': {'min_root_height_m': .6, 'max_mean_joint_rmse_rad': .35},
    'admission': 'Every proposal must pass current asset, continuous joint-limit, strict future-input and execution checks.',
    'limitations': ['Sabit geçiş saati ritim eşleme değildir.',
                    'Kol RMS oranı algısal ifade veya insan beğenisi ölçümü değildir.',
                    'Hazır iki referansı düzenler; yeni dans becerisi öğrenmez.'],
}


def tool_catalog():
    return [deepcopy(_TOOL)]
