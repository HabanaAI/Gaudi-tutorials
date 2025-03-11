# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import unittest

import requests

unittest.TestLoader.sortTestMethodsUsing = None


# unittest.TestLoader.sortTestMethodsUsing = lambda self, a, b: (a < b) - (a > b)
class RunCmd:
    def run(self, cmd):
        import subprocess
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        # print("Command exit status/return code : ", p_status)
        return p_status, output

class PerfUtility:

    def __init__(self, filename):
        self.class_name = filename.split(".")[0]
        self.filename = filename  # self.__class__.__name__
        return

    def load_input_data(self, model_name):
        import os
        gaudi_version = os.getenv('GAUDI_VER') or '3'
        class_name = self.class_name + gaudi_version
        matched_items=[]
        with open(self.filename, "r") as file:
            try:
                data = json.load(file)
            except:
                print("json load failed: " + self.filename)
                pass
        for i in data[class_name]:
            if i["model"] == model_name:
                matched_items.append(i)
        return matched_items

    def model_test(self, data, perf_report):
        # 0. setup run env
        import os
        hqt_output_exist = False
        src = "HQT" + os.sep + "hqt_output_" + data["model"] + '_' + data["num_cards"] + 'c'
        dst = "hqt_output"
        if os.path.islink(dst):
            os.unlink(dst)
        if os.path.exists(src):
            os.symlink(src,dst)
            hqt_output_exist = True
        else:
            print("No Tensor measurement output, Need redo the Tensor Measurement")

        ref_perf = data["ref_perf"]
        if ref_perf =="":
            status = 0
            if hqt_output_exist is False:
                # 0.1 .Run the tensor measurement instruction
                import shutil
                import os
                cmd = data["run_cmd"]
                status, output = RunCmd().run(cmd)
                output = output.decode("utf-8")
                print(cmd)
                # 0.1.1 copy generated hqt_output(dst) to HQT folder(src)
                if os.path.exists(dst):
                    if not os.path.exists(src):
                        os.makedirs(src)
                    file_names = os.listdir(dst)
                    for file_name in file_names:
                        shutil.move(os.path.join(dst, file_name), src)
                    os.rmdir(dst)
                # end 0.1 Tensor measurement
            return status

        # 1.Run the instruction
        cmd = data["run_cmd"]
        #print(cmd)
        #return 0
        status, output = RunCmd().run(cmd)
        output = output.decode("utf-8")

        # 2.Parsing the run log
        filename = data["model"] + "_" + data["input_len"] + "_" + data["output_len"] + "_" + data["num_cards"] + 'c' + "_log.txt"
        perf_report.dump_log_to_file(output, filename)
        throughput, mem_allocated, max_mem_allocated, graph_compile = perf_report.parse_run_log(output)

        # 3.Add new row into report
        #throughput = '0'
        new_row = {}
        perf_ratio = float(throughput) / float(data["ref_perf"])
        if perf_report.report_level >= 3:
            new_row = {"Model": data["model"], "#cards": data["num_cards"], "InputLen": data["input_len"], "OutputLen": data["output_len"], "BS": data["bs"], "ref_perf_number": data["ref_perf"], "perf_number": throughput, "perf_ratio": perf_ratio, "max_mem_allocated": max_mem_allocated ,"cmd": data["run_cmd"]}
        else:
            new_row = {"Model": data["model"], "#cards": data["num_cards"], "InputLen": data["input_len"], "OutputLen": data["output_len"], "BS": data["bs"], "ref_perf_number": data["ref_perf"], "perf_number": throughput, "perf_ratio": perf_ratio, "max_mem_allocated": max_mem_allocated}

        df_len = len(perf_report.perf_report_df)
        perf_report.perf_report_df.loc[df_len+1] = new_row
        return status


class PerfReport:
    def __init__(self, name, report_level):
        self.name = name
        self.report_level = report_level
        self.env_vars_df = None
        self.system_info_df = None
        self.gaudi_info_df = None
        self.docker_ps = ""
        self.docker_ps_df = None
        self.perf_report_df = None
        import datetime

        d = datetime.datetime.now()
        dateinfo = d.strftime("%m-%d_%H-%M")
        self.result_folder_name = self.name + "_" + dateinfo
        import os

        if not os.path.exists(self.result_folder_name):
            os.makedirs(self.result_folder_name)


    def init_perf_report(self):

        import pandas as pd
        rows = []
        if report_level >= 3:
            columns = ["Model", "#cards", "InputLen", "OutputLen", "BS", "ref_perf_number", "perf_number", "perf_ratio", "max_mem_allocated", "cmd"]
        else:
            columns = ["Model", "#cards", "InputLen", "OutputLen", "BS", "ref_perf_number", "perf_number", "perf_ratio", "max_mem_allocated"]

        df = pd.DataFrame(rows, columns=columns)
        self.perf_report_df = df

    def dump_log_to_file(self, output, filename):
        filepath = self.result_folder_name + os.sep + filename
        fd = open(filepath, "w")  # append mode
        fd.write(output)
        fd.close()
        return

    def parse_run_log(self, log):
        throughput = ''
        mem_allocated = ''
        max_mem_allocated = ''
        graph_compile = ''
        for line in log.splitlines():
            if line.find("Throughput") != -1:
                throughput = line.split('=')[1].split(' ')[1]
            elif line.find("Memory") != -1:
                mem_allocated = line.split('=')[1].split(' ')[1]
            elif line.find("Max") != -1:
                max_mem_allocated = line.split('=')[1].split(' ')[1]
            elif line.find("Graph") != -1:
                graph_compile = line.split('=')[1].split(' ')[1]
        return throughput, mem_allocated, max_mem_allocated, graph_compile


    def generate_perf_report(self):
        import os
        import re

        print(" Example Name:" + self.name)
        print(" ### System Info###")
        print(self.system_info_df)
        print(" ### Gaudi Info###")
        print(self.gaudi_info_df)
        self.docker_ps_df = None
        print(" ### Performance Number###")
        print(self.perf_report_df)

        report_name = self.name + ".html"

        report_path = self.result_folder_name + os.sep + report_name

        # Log Files

        docker_log_html_content = ""
        pattern = r".*\_docker_log.txt$"  # Match all files ending with ".txt"
        for filename in os.listdir(self.result_folder_name):
            if re.search(pattern, filename):
                html_content = (
                    " \n\n <h2>"
                    + filename
                    + "</h2>\n"
                    + "<iframe src="
                    + '"'
                    + filename
                    + '"'
                    + " width="
                    + '"'
                    + "1000"
                    + '"'
                    + "height="
                    + '"'
                    + "300"
                    + '"'
                    + "></iframe>"
                )
                docker_log_html_content = docker_log_html_content + html_content

        with open(report_path, "w") as hfile:
            hfile.write(
                "\n\n <h1>1. Perf Numbers</h1>\n\n"
                + self.perf_report_df.to_html()
                + "\n\n <h1>2. System Info</h1>\n\n"
                + self.system_info_df.head().to_html()
                + "\n\n <h1>3. Gaudi Info</h1>\n\n"
                + self.gaudi_info_df.head().to_html()
            )

        print("\nReport File is : " + report_path)
        import shutil

        shutil.make_archive(self.result_folder_name, "zip", self.result_folder_name)
        return


class OH_Benchmark(unittest.TestCase):
    skip_llama2_70b=int(os.environ.get('skip_llama2_70b', 0))
    skip_llama31_8b=int(os.environ.get('skip_llama31_8b', 0))
    skip_llama31_70b=int(os.environ.get('skip_llama31_70b', 0))
    skip_llama33_70b=int(os.environ.get('skip_llama33_70b', 0))
    skip_llama31_405b=int(os.environ.get('skip_llama31_405b', 0))
    def setUp(self):
        self.perf_report = perf_report
        self.ip = "http://0.0.0.0"
        self.datafile = DataJsonFileName
        self.classname = DataJsonFileName.split(".")[0]
        self.utils = PerfUtility(self.datafile)
        self.hostname = ''
        if not os.path.exists("./HQT") and os.path.exists("./HQT.zip"):
            import zipfile
            zip = zipfile.ZipFile('HQT.zip')  # from zipfile import ZipFile
            zip.extractall('./')
            zip.close()
        return

    def tearDown(self):
        return

    def test_0_system(self):

        import socket

        self.hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(self.hostname)

        import platform

        system_info = platform.uname()
        import pandas as pd

        rows = []
        columns = ["info", "value"]
        rows.append(["hostname", self.hostname])
        rows.append(["ip", IPAddr])
        rows.append(["system", system_info.system])
        rows.append(["node", system_info.node])
        rows.append(["release", system_info.release])
        rows.append(["version", system_info.version])
        rows.append(["machine", system_info.machine])
        rows.append(["processor", system_info.processor])
        df = pd.DataFrame(rows, columns=columns)
        self.perf_report.system_info_df = df

        self.perf_report.init_perf_report()

        self.assertEqual(False, False)
    
    def test_1_perfspect(self):
        # PerfSpect Report
        if not os.path.exists("./perfspect"):
            cmd = 'wget -qO- https://github.com/intel/PerfSpect/releases/latest/download/perfspect.tgz | tar xvz'
            status, output = RunCmd().run(cmd)
        cmd = './perfspect/perfspect report --gaudi --output ' + self.perf_report.result_folder_name
        status, output = RunCmd().run(cmd)
        import socket
        hostname = socket.gethostname()
        xlsx_file = self.perf_report.result_folder_name + os.sep + hostname + '.xlsx'
        import pandas as pd
        if os.path.exists(xlsx_file):
            print(xlsx_file)
            df= pd.read_excel(xlsx_file)
            print(df)
            self.perf_report.gaudi_info_df = df
        self.assertEqual(False, False)

    @unittest.skipIf(skip_llama2_70b == 1 , "Skip over this routine")
    def test_2_llama2_70b(self):

        model_name = "Llama2_70b"
        # Get configs/data
        data = self.utils.load_input_data(model_name)
        #print(data)
        self.assertNotEqual(data, None)

        # Testing
        for i in data:
            response_status_code = self.utils.model_test(i, perf_report)
        self.assertEqual(response_status_code, 0)

    @unittest.skipIf(skip_llama31_8b == 1 , "Skip over this routine")
    def test_3_llama3_1_8b(self):

        model_name = "Llama3.1_8b"
        # Get configs/data
        data = self.utils.load_input_data(model_name)
        #print(data)
        self.assertNotEqual(data, None)

        # Testing
        for i in data:
            response_status_code = self.utils.model_test(i, perf_report)
        self.assertEqual(response_status_code, 0)

    @unittest.skipIf(skip_llama31_70b == 1 , "Skip over this routine")
    def test_4_llama3_1_70b(self):

        model_name = "Llama3.1_70b"
        # Get configs/data
        data = self.utils.load_input_data(model_name)
        #print(data)
        self.assertNotEqual(data, None)

        # Testing
        for i in data:
            response_status_code = self.utils.model_test(i, perf_report)
        self.assertEqual(response_status_code, 0)

    @unittest.skipIf(skip_llama33_70b == 1 , "Skip over this routine")
    def test_5_llama3_3_70b(self):

        model_name = "Llama3.3_70b"
        # Get configs/data
        data = self.utils.load_input_data(model_name)
        #print(data)
        self.assertNotEqual(data, None)

        # Testing
        for i in data:
            response_status_code = self.utils.model_test(i, perf_report)
        self.assertEqual(response_status_code, 0)

    @unittest.skipIf(skip_llama31_405b == 1 , "Skip over this routine")
    def test_6_llama3_1_405b(self):

        model_name = "Llama3.1_405b"
        # Get configs/data
        data = self.utils.load_input_data(model_name)
        #print(data)
        self.assertNotEqual(data, None)

        # Testing
        for i in data:
            response_status_code = self.utils.model_test(i, perf_report)
        self.assertEqual(response_status_code, 0)

if __name__ == "__main__":
    import sys
    import os

    report_level = 2  # low, medium, high
    DataJsonFileName = "Gaudi.json" #sys.argv[1]  # "ChatQnA_Xeon.json"
    if os.path.isfile(DataJsonFileName) is False:
        print("Missing Gaudi.json file")
        exit(0)

    perf_report = PerfReport(DataJsonFileName, report_level)
    test_loader = unittest.TestLoader()
    suite = test_loader.loadTestsFromTestCase(OH_Benchmark)
    unittest.TextTestRunner(verbosity=3).run(suite)
    perf_report.generate_perf_report()
