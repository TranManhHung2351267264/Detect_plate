"""Restore runnable demo models for local webapp and check_app testing."""
import json
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import joblib
import numpy as np

from plate_ml.config import DetectionConfig, HOGConfig, SegmentationConfig
from plate_ml.domain_ocr import extract_vnlp_glyphs
from plate_ml.features import HOGExtractor
from plate_ml.glyph_quality import GlyphQualityConfig
from plate_ml.hmm import VietnamesePlateHMM
from plate_ml.models import make_svm, PCA_FLD_Gaussian
from plate_ml.pipeline import PlateRecognizer
from plate_ml.registry import ModelRegistry, file_sha256
from plate_ml.research_pipeline import RegionConfig, train_region_verifier
from plate_ml.vnlp import scan_vnlp

print("=" * 65)
print("  KHỞI TẠO MÔ HÌNH VÀ TÀI NGUYÊN PHỤC VỤ WEBAPP (LOCAL DEMO)")
print("=" * 65)

# 1. Quét dữ liệu VNLP
vnlp_root = Path(r"d:\Nhận Dạng Mẫu\DOAN_Nhandienbiensoxe\VNLP_detection\detection")
print(f"[*] Quét ảnh VNLP từ: {vnlp_root}")
scenes, _ = scan_vnlp(vnlp_root, max_per_layout=30, seed=42, box_format="xyxy")
print(f"    Tổng số ảnh phương tiện: {len(scenes)}")

# Sao chép 1 ảnh làm mẫu mặc định multi_dectec_sample.jpg nếu chưa có
sample_path = ROOT / "multi_dectec_sample.jpg"
if not sample_path.exists() and scenes:
    shutil.copyfile(scenes[0].path, sample_path)
    print(f"    -> Đã tạo ảnh mẫu mặc định: {sample_path.name}")

# 2. Trích xuất glyphs
print("[*] Trích xuất ký tự và đặc trưng HOG...")
glyphs = extract_vnlp_glyphs(scenes)
X, y = glyphs.X, glyphs.y
unique, counts = np.unique(y, return_counts=True)
valid_classes = set(unique[counts >= 2])
mask = np.isin(y, list(valid_classes))
X, y = X[mask], y[mask]
print(f"    Số mẫu ký tự: {len(X)} | Số lớp: {len(valid_classes)}")

# 3. Huấn luyện SVM
print("[*] Huấn luyện SVM OCR...")
svm = make_svm(c=3.0, kernel="rbf", seed=42)
svm.fit(X, y)

# 4. Huấn luyện PCA + FLD + Gaussian
print("[*] Huấn luyện PCA + FLD + Gaussian Emission...")
pca_components = min(48, len(X) - len(valid_classes), X.shape[1])
pca_fld_gaussian = PCA_FLD_Gaussian(pca_components=pca_components, covariance_regularization=1e-6)
pca_fld_gaussian.fit(X, y)

# 5. Huấn luyện HMM
print("[*] Huấn luyện HMM Bigram...")
plate_texts = [s.text for s in scenes if s.text]
hmm = VietnamesePlateHMM(smoothing=1.0).fit(plate_texts)

# 6. Huấn luyện Region Verifier
print("[*] Huấn luyện Region Verifier cho Two-Stage Detector...")
verifier, _ = train_region_verifier(scenes[:30], seed=42)

# 7. Đóng gói Baseline & Research Candidate Bundle
cascade_xml = ROOT / "assets/cascades/haarcascade_russian_plate_number.xml"
detection_cfg = DetectionConfig(cascade_path=str(cascade_xml))
baseline = PlateRecognizer(
    ocr_model=svm,
    detection=detection_cfg,
    segmentation=SegmentationConfig(),
    hog=HOGConfig(),
    verifier=verifier,
)

selection_cfg = {
    "detector": "new",
    "segmenter": "hybrid_seg",
    "threshold": -0.5,
    "rerank": 0.5,
    "top_k": 5,
}

candidate_bundle = dict(
    schema="research_pipeline_v1",
    baseline_bundle=baseline.to_bundle(),
    verifier=verifier,
    selection_config=selection_cfg,
)

# 8. Đăng ký vào Registration/Registry/research_glyph_filter/v1
print("[*] Đăng ký bản phát hành research_glyph_filter/v1...")
registry_dir = ROOT / "Registration"
registry = ModelRegistry(registry_dir)
release_v1_dir = registry_dir / "Registry/research_glyph_filter/v1"
release_v1_dir.mkdir(parents=True, exist_ok=True)

model_payload = dict(
    base_bundle=candidate_bundle,
    quality_config=asdict(GlyphQualityConfig()),
)
model_file = release_v1_dir / "model.joblib"
joblib.dump(model_payload, model_file, compress=3)
model_sha = file_sha256(model_file)

meta = {
    "name": "research_glyph_filter",
    "version": "v1",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "artifact_sha256": {
        "model.joblib": model_sha,
    },
}
(release_v1_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"    -> Đã ghi: {release_v1_dir}")

# 9. Đóng gói PCA+FLD+HMM candidate vào reports/pca-fld-hmm-no-svm-180-20261026/
print("[*] Đóng gói no_svm_candidate.joblib...")
reports_dir = ROOT / "reports/pca-fld-hmm-no-svm-180-20261026"
reports_dir.mkdir(parents=True, exist_ok=True)
no_svm_file = reports_dir / "no_svm_candidate.joblib"
joblib.dump(dict(model=pca_fld_gaussian, hmm=hmm), no_svm_file, compress=3)
no_svm_sha = file_sha256(no_svm_file)

proto = {
    "candidate_sha256": no_svm_sha,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(reports_dir / "protocol_before_test.json").write_text(json.dumps(proto, indent=2), encoding="utf-8")
print(f"    -> Đã ghi: {reports_dir}")

print("\n" + "=" * 65)
print("  HOÀN TẤT PHỤC HỒI MÔ HÌNH! SẴN SÀNG CHO WEBAPP VÀ DEMO!")
print("=" * 65)
