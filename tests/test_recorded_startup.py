"""Clean-checkout delivery and startup fallback, without ROS or network."""
import hashlib
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DIGEST = 'b33b2debe3c1710bd6e7458b983767bfc7a7ce003b0d9c5560ad20ec5bd7fade'


class RecordedStartupTests(unittest.TestCase):
    def test_pose_fitting_follows_selected_families_and_preserves_override(self):
        from hack_motion.recorded_grasp import PROFILES
        tree = ast.parse((ROOT/'src/hack_bringup/launch/solution.launch.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'recorded_pose_names')
        class Configuration:
            def __init__(self, name): self.name = name
            def perform(self, context): return context[self.name]
        scope = dict(LaunchConfiguration=Configuration, SetLaunchConfiguration=lambda k,v: (k,v))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<launch>', 'exec'), scope)
        resolve = scope['recorded_pose_names']
        params = dict(recorded_grasps='true', recorded_objects='philips', rgbd_model_names='')
        self.assertEqual(resolve(params), [('rgbd_model_names', next(p['object_name'] for p in PROFILES.values() if p['family']=='philips'))])
        params['recorded_objects']='brother, philips'
        names=resolve(params)[0][1].split(',')
        self.assertEqual(set(names), {p['object_name'] for p in PROFILES.values() if p['family'] in ('brother','philips')})
        self.assertEqual(resolve(dict(params, rgbd_model_names='explicit')), [])
        self.assertEqual(resolve(dict(params, recorded_grasps='false')), [])

    def test_clean_checkout_stages_bundled_archive_with_pinned_digest(self):
        archive = ROOT/'assets/perception/geometry_only_ros_bundle.tar.gz'
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), DIGEST)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'tools').mkdir(); (root/'assets/perception').mkdir(parents=True)
            shutil.copy(ROOT/'tools/stage_perception_rgbd.sh', root/'tools')
            (root/'assets/perception/geometry_only_ros_bundle.tar.gz').symlink_to(archive)
            result = subprocess.run(['bash','tools/stage_perception_rgbd.sh'], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            bundle = root/'weights'/('rgbd-'+DIGEST[:12])
            manifest = json.loads((bundle/'bundle.json').read_text())
            for name, digest in manifest['sha256'].items():
                self.assertEqual(hashlib.sha256((bundle/name).read_bytes()).hexdigest(), digest)
            # An explicit corrupted override is rejected, never silently bypassed.
            (root/'weights/geometry_only_ros_bundle.tar.gz').write_bytes(b'bad')
            result = subprocess.run(['bash','tools/stage_perception_rgbd.sh'], cwd=root, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_startup_requires_ready_marker_and_manifest_and_honours_optout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); shutil.copy(ROOT/'run.sh',root)
            (root/'bin').mkdir()
            ros = root/'bin/ros2'
            ros.write_text('#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps({"env":{k:v for k,v in os.environ.items() if k.startswith("NSSR_")},"args":sys.argv[1:]}))\n')
            ros.chmod(0o755)
            env = {k:v for k,v in os.environ.items() if not k.startswith('NSSR_')}
            env['PATH'] = str(root/'bin')+os.pathsep+env['PATH']
            def launch(**extra):
                p = subprocess.run(['bash','run.sh'],cwd=root,env=dict(env,**extra),capture_output=True,text=True,check=True)
                return json.loads(p.stdout)['env']
            self.assertEqual(launch()['NSSR_RECORDED_GRASPS'],'false')
            (root/'weights').mkdir(); (root/'weights/.rgbd-ready').touch()
            self.assertEqual(launch()['NSSR_RECORDED_GRASPS'],'false')
            bundle = root/'weights'/('rgbd-'+DIGEST[:12]);bundle.mkdir();(bundle/'bundle.json').write_text('{}')
            result=launch()
            self.assertEqual(result['NSSR_RECORDED_GRASPS'],'false')
            self.assertEqual(result['NSSR_RECORDED_OBJECTS'],'philips')
            self.assertEqual(result['NSSR_RGBD_ON_DEMAND'],'true')
            self.assertEqual(launch(NSSR_RECORDED_GRASPS='true')['NSSR_RECORDED_GRASPS'],'true')
            self.assertEqual(launch(NSSR_RECORDED_GRASPS='false')['NSSR_RECORDED_GRASPS'],'false')
            self.assertEqual(launch(NSSR_RECORDED_OBJECTS='brother')['NSSR_RECORDED_OBJECTS'],'brother')

    def test_failed_setup_returns_zero_and_clears_previous_ready_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shutil.copy(ROOT/'setup.sh',root)
            (root/'tools').mkdir();(root/'weights').mkdir();(root/'weights/.rgbd-ready').touch()
            for name in ('curobo_install.sh','stage_perception_rgbd.sh'):
                (root/'tools'/name).write_text('exit 1\n')
            result=subprocess.run(['bash','setup.sh'],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((root/'weights/.rgbd-ready').exists())
            self.assertIn('using baseline perception',result.stdout)
