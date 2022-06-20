# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from inspect import getfile, isclass
from mock import patch, Mock
import sys
import unittest
from sagemaker_training import errors

#
class PseudoPackage:
    def __init__(self, exceptions=None):
        if exceptions:
            self.exceptions = exceptions


class PseudoExceptionFile:
    def __init__(self, type="backend"):
        self.type = type
        self.PseudoBackendException = PseudoBackendException
        self.PseudoTorchException = PseudoTorchException

    def __dir__(self):
        return ["PseudoBackendException"] if self.type == "backend" else ["PseudoTorchException"]


class PseudoTorchException:
    def __init__(self):
        return


class PseudoBackendException:
    def __init__(self):
        return


@patch.dict(sys.modules, {"smdistributed": Mock()})
@patch.dict(sys.modules, {"smdistributed.modelparallel": Mock()})
@patch.dict(
    sys.modules,
    {"smdistributed.modelparallel.backend": PseudoPackage(PseudoExceptionFile("backend"))},
)
@patch.dict(
    sys.modules, {"smdistributed.modelparallel.torch": PseudoPackage(PseudoExceptionFile("torch"))}
)
def test_smp_import():
    from sagemaker_training import mpi as mpi

    exceptions = mpi.exception_classes
    assert exceptions == ["PseudoBackendException", "PseudoTorchException"]


# @patch.dict(sys.modules, {'smdistributed': Mock()})
# @patch.dict(sys.modules, {'smdistributed.modelparallel': Mock()})
# @patch.dict(sys.modules, {'smdistributed.modelparallel.backend': PseudoPackage()})
# @patch.dict(sys.modules, {'smdistributed.modelparallel.torch': PseudoPackage(PseudoExceptionFile('torch'))})
# def test_smp_import_torch_oonly():

#     from sagemaker_training import mpi as mpi

#     exceptions = mpi.exception_classes
#     assert exceptions == ['PseudoTorchException']


@patch.dict(sys.modules, {"smdistributed": Mock()})
@patch.dict(sys.modules, {"smdistributed.modelparallel": Mock()})
@patch.dict(sys.modules, {"smdistributed.modelparallel.backend": PseudoPackage()})
@patch.dict(sys.modules, {"smdistributed.modelparallel.torch": PseudoPackage()})
def test_smp_import_no_exceptions():

    from sagemaker_training import mpi as mpi_2

    exceptions = mpi_2.exception_classes
    assert exceptions == [errors.ExecuteUserScriptError], f"exceptions are {exceptions}"


# sys.modules['smdistributed.modelparallel.backend.excetions'] = abx_ob


if __name__ == "__main__":
    # unittest.main()
    test_smp_import()
    test_smp_import_no_exceptions()
    # test_smp_import_torch_oonly()
# sys.modules['smdistributed.modelparallel.backend.excetions'] = None
# exception_classes = []
# from smdistributed.modelparallel.backend import exceptions
# exception_classes += [x for x in dir(exceptions) if isclass(getattr(exceptions, x))]
# print(exception_classes)
# mock.patch.dict(
#     sys.modules,
#     {'smdistributed.modelparallel.backend.exceptions': mock_exp},
#
