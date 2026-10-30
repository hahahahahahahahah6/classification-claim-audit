import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from classification_claim_audit.audit import audit, claims, evidence, matches
from classification_claim_audit.phrases import CLASSIFICATION, NOCLASSIFICATION
from classification_claim_audit.__main__ import main

FIXTURES = Path(__file__).parent / 'fixtures'


class ClaimsTests(unittest.TestCase):
    def test_counts_and_sort(self):
        self.assertEqual(len(CLASSIFICATION), 52)
        self.assertEqual(len(NOCLASSIFICATION), 36)
        for values in (CLASSIFICATION, NOCLASSIFICATION):
            self.assertEqual(len(values), len(set(values)))
            self.assertEqual(list(values), sorted(values, key=len, reverse=True))

    def test_fences(self):
        for fence in ('```', '~~~', '   ````'):
            self.assertEqual(claims(fence+'\nclassifies images\n'+fence), (False, False))

    def test_inline(self):
        self.assertEqual(claims('`classifies images`'), (True, False))

    def test_normalization(self):
        self.assertEqual(claims("Doesn't classify images into categories"), (False, True))
        self.assertEqual(claims('REAL-TIME IMAGE\nCLASSIFICATION'), (True, False))
        self.assertEqual(claims('classifies\n\nimages'), (False, False))

    def test_longest_and_boundaries(self):
        self.assertEqual(matches('classifies images into categories', reversed(CLASSIFICATION))[0], ['classifies images into categories'])
        self.assertEqual(claims('preclassify imagesx'), (False, False))


POSITIVES = ['import torchvision\ntorchvision.models.resnet18()', 'import torchvision as tv\ntv.models.vgg16()', 'from torchvision.models import efficientnet_b0 as model\nmodel()', 'import timm\ntimm.create_model("x")', 'from timm import create_model as m\nm("x")', 'import keras\nkeras.applications.ResNet50()', 'import tensorflow as tf\ntf.keras.applications.MobileNetV2()', 'from keras.applications import Xception as X\nX()', 'from transformers import pipeline as p\np("image-classification")', 'import transformers as t\nt.pipeline(task="image-classification")']
NEGATIVES = ['import torchvision', 'import cv2\ncv2.imread("x")', 'from PIL import Image\nImage.open("x")', 'import cv2\ncv2.VideoCapture(0)', 'import cv2\ncv2.rectangle(x,a,b,c)', 'from torchvision.models.detection import fasterrcnn_resnet50_fpn\nfasterrcnn_resnet50_fpn()', 'from ultralytics import YOLO\nYOLO("x")', 'import mediapipe as mp\nmp.solutions.selfie_segmentation.SelfieSegmentation()', 'import mediapipe as mp\nmp.tasks.vision.ImageSegmenter.create_from_options(o)', 'import face_recognition\nface_recognition.face_locations(x)', 'import mediapipe as mp\nmp.solutions.hands.Hands()', 'import mediapipe as mp\nmp.solutions.pose.Pose()', 'import pytesseract\npytesseract.image_to_string(x)', 'from pyzbar.pyzbar import decode\ndecode(x)', 'model.track(x)', 'from sklearn.linear_model import LogisticRegression\nLogisticRegression().fit(x,y)', 'import torch\ntorch.nn.Linear(2,3)', 'import cv2\nn=cv2.dnn.readNet("x")\nn.forward()', 'import torch\ntorch.hub.load("x","resnet18")', 'from torchvision.models import get_model\nget_model("resnet18")', 'from timm import create_model\ndef create_model(x): return x\ncreate_model("x")', 'from transformers import pipeline\nclass pipeline: pass\npipeline("image-classification")', 'from torchvision.models import resnet18\nresnet18=lambda:None\nresnet18()', 'resnet18()\ncreate_model("x")\npipeline("image-classification")', 'x="classification"', 'import subprocess\nsubprocess.run(["classify"])', 'not valid python !', 'from transformers import pipeline\npipeline("object-detection")', 'import open_clip\nopen_clip.create_model("x")', 'from other import create_model\ncreate_model("x")']

class EvidenceTests(unittest.TestCase):
    def test_all_listed_constructors(self):
        from classification_claim_audit.audit import MODELS, KERAS_MODELS
        for name in MODELS:
            with self.subTest(model=name):
                self.assertTrue(evidence('from torchvision.models import '+name+' as m\nm()'))
        for name in KERAS_MODELS:
            with self.subTest(model=name):
                self.assertTrue(evidence('from tensorflow.keras.applications import '+name+' as m\nm()'))

    def test_parameter_shadows(self):
        for parameter in ('pipeline', '*pipeline', '**pipeline'):
            self.assertFalse(evidence('from transformers import pipeline\ndef f('+parameter+'):\n pipeline("image-classification")'))



class IntegrationTests(unittest.TestCase):
    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                status=main(list(map(str,args)))
            except SystemExit as e:
                status=e.code
        return status,out.getvalue(),err.getvalue()

    def test_fixtures(self):
        for fixture, status, severity, rule in [
            ('good',0,None,None), ('no_claim',0,None,None),
            ('bad',2,'HIGH','documented-classification-claim-false'),
            ('testonly',0,'MEDIUM','documented-classification-claim-false'),
            ('no_classification',2,'HIGH','documented-no-classification-claim-false')]:
            with self.subTest(fixture=fixture):
                code,out,_=self.cli(FIXTURES/fixture,'--json')
                self.assertEqual(code,status)
                result=json.loads(out)
                self.assertEqual(set(result),{'verdict','findings','files_scanned'})
                self.assertEqual(result['verdict'],'FAIL' if status==2 else 'PASS')
                self.assertEqual(len(result['findings']),int(severity is not None))
                if severity:
                    self.assertEqual(result['findings'][0]['severity'],severity)
                    self.assertEqual(result['findings'][0]['rule'],rule)
                    self.assertEqual(set(result['findings'][0]),{'rule','severity','message'})
        self.assertEqual(self.cli(FIXTURES/'testonly','--fail-on','MEDIUM')[0],2)

    def test_ignore_and_verbose(self):
        code,out,_=self.cli(FIXTURES/'bad','--ignore','README.md','--verbose')
        self.assertEqual(code,0)
        self.assertIn('Files scanned: 1',out)
        self.assertEqual(self.cli(FIXTURES/'good','--ignore','*.py')[0],2)

    def test_version_errors(self):
        self.assertEqual(self.cli('--version')[:2],(0,'0.1.0\n'))
        for args in [[],['--unknown'],['/nonexistent-root'],[str(FIXTURES/'bad'),'--fail-on','LOW']]:
            self.assertEqual(self.cli(*args)[0],1)

    def test_exclusions_and_docs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'README.md').write_text('classifies images')
            for component in ['docs','examples','fixtures']:
                folder=root/component;folder.mkdir();(folder/'source.py').write_text(POSITIVES[0])
            (root/'linked.py').symlink_to(root/'docs/source.py')
            (root/'linked_dir').symlink_to(root/'docs',target_is_directory=True)
            self.assertEqual(audit(root)['findings'][0]['severity'],'HIGH')
            (root/'README.md').write_text('Nothing promised.')
            (root/'docs/note.md').write_text('classifies images')
            self.assertEqual(audit(root)['verdict'],'FAIL')

    def test_self_audit(self):
        self.assertEqual(audit(Path(__file__).resolve().parents[1])['verdict'],'PASS')


def add_test(cls, name, function, value, expected):
    def test(self):
        self.assertEqual(function(value), expected)
    setattr(cls, name, test)


for i,phrase in enumerate(CLASSIFICATION):
    add_test(ClaimsTests,'test_positive_%02d'%i,claims,phrase,(True,False))
for i,phrase in enumerate(NOCLASSIFICATION):
    add_test(ClaimsTests,'test_negative_%02d'%i,claims,phrase,(False,True))
for i,source in enumerate(POSITIVES):
    add_test(EvidenceTests,'test_provider_%02d'%i,evidence,source,True)
for i,source in enumerate(NEGATIVES):
    add_test(EvidenceTests,'test_nonprovider_%02d'%i,evidence,source,False)
