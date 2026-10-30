# Whole-picture labelling audit

Offline static checking of documented whole-picture labelling promises against Python provider evidence. Python 3.9 or newer; MIT licensed; no runtime dependencies. Source is parsed and never executed.

```sh
pip install classification-claim-audit
classification-claim-audit ./project --json
python3 -m classification_claim_audit ./project --verbose
classification-claim-audit ./project --ignore 'vendor/*' --fail-on MEDIUM
classification-claim-audit --version
```

Reads root README.md and Markdown files beneath any docs path component. Backtick and tilde fenced blocks with up to three leading spaces are excluded; inline code counts. Blank lines separate paragraphs, wrapped lines join, case is folded, contractions expand, hyphens become spaces, and whitespace collapses. Exact templates match word boundaries, longest first. Negative spans are removed before positive matching.

Evidence requires an alias-resolved call in a single module. Local definitions shadow imports. Construction alone counts for the first three providers; the fourth requires the literal task shown:

```text
torchvision.models: listed constructors, including resnet18, vgg16, efficientnet_b0
timm.create_model
keras.applications / tensorflow.keras.applications: listed constructors, including ResNet50, MobileNetV2
transformers.pipeline("image-classification")
transformers.pipeline(task="image-classification")
```

Complete constructor sets and wording templates are in the package modules. Bare imports, picture opening, camera capture, drawing supplied boxes, object finding, pixel segmentation, face/hand/pose detection, OCR, barcode decoding, tracking, generic tabular estimators, and bare neural heads do not count. Neither do strings, unbound calls, subprocess helpers, unrelated providers, or syntax-invalid files.

Symlinked files and directories are excluded. Python files under fixtures, docs, and examples are excluded. Files under tests or named test_*.py or *_test.py provide only test evidence. Repeatable ignore globs use fnmatch on relative POSIX paths.

At most one finding appears per family:

```text
documented-classification-claim-false
  HIGH: positive promise lacks production evidence
  MEDIUM: evidence exists only in tests
documented-no-classification-claim-false
  HIGH: negative promise conflicts with production evidence
```

No matching promise means PASS regardless of source. The default threshold is HIGH; MEDIUM also fails on test-only evidence. Exits are 0 for PASS, 2 for FAIL, and 1 for invalid or unreadable roots and usage errors. Human output includes verdict and findings; verbose output adds the scanned count. JSON includes verdict, findings (rule, severity, message), and files_scanned.

Static construction evidence cannot establish runtime success, prediction quality, weights, or actual use. Dynamic imports, cross-module flow, custom implementations, text-to-picture similarity, and runtime model names are outside scope. These operations do not count:

```text
CLIP / open_clip zero-shot matching
cv2.dnn.readNet / forward
torch.hub.load
torchvision.models.get_model("resnet18")
```

Develop and build with:

```sh
python3 -m unittest discover
python3 -m build
```
