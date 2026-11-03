from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import task07_fastcheck as F


class FastCheckTests(unittest.TestCase):
    def test_analysis_uses_public_image_and_read_only_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'result'
            def run(command,**kwargs):
                (out/'contracts.json').write_text('{"reports":[]}')
                return type('Done',(),{'stdout':'','stderr':''})()
            with patch.object(F.P,'checked',side_effect=run) as checked:
                F.check(root,out)
            command=checked.call_args.args[0]
            self.assertIn(F.Q.AGENT_IMAGE,command)
            self.assertNotIn(F.Q.IMAGE,command)
            self.assertIn(str(root.resolve())+':/cases:ro',command)
            self.assertFalse(any('/tests' in part for part in command))

    def test_timeout_removes_only_its_owned_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(F.P,'checked',side_effect=subprocess.TimeoutExpired('docker',1)) as checked,patch.object(F.subprocess,'run') as stop:
                with self.assertRaises(TimeoutError):F.check(root,root/'out',1)
            command=checked.call_args.args[0]
            name=command[command.index('--name')+1]
            self.assertEqual(stop.call_args.args[0],['docker','rm','-f',name])


if __name__=='__main__':unittest.main()
