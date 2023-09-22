"""
Dynex SDK (beta) Neuromorphic Computing Library
Copyright (c) 2021-2023, Dynex Developers

All rights reserved.

1. Redistributions of source code must retain the above copyright notice, this list of
    conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice, this list
   of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors may be
   used to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__version__ = "0.1.5"
__author__ = 'Dynex Developers'
__credits__ = 'Dynex Developers, Contributors, Supporters and the Dynex Community'

################################################################################################################################
# IMPORTS
################################################################################################################################

# Encryption:
from Crypto.Cipher import AES
import base64
import binascii
import hashlib
import secrets

# required libs:
from pathlib import Path
from ftplib import FTP
import dimod
from itertools import combinations
import time
import numpy as np

# ini config reader:
import configparser

# progress information:
from IPython.display import clear_output
from tabulate import tabulate
from tqdm.notebook import tqdm

# test-net mode:
import subprocess
import os
import sys

# API functions:
import urllib.request, json
import base64

################################################################################################################################
# API FUNCTION CALLS
################################################################################################################################

FILE_IV = '';
FILE_KEY = '';
MAX_CHIPS = 0;
MAX_ANNEALING_TIME = 0;
MAX_DURATION = 0;
TOTAL_USAGE = False;

# parse config file:
try:
	config = configparser.ConfigParser();
	config.read('dynex.ini', encoding='UTF-8');
	API_ENDPOINT = config['DYNEX']['API_ENDPOINT']
	API_KEY = config['DYNEX']['API_KEY'];
	API_SECRET = config['DYNEX']['API_SECRET'];
except:
	raise Exception('ERROR: missing configuration file dynex.ini');

def account_status():
    """
    Shows the status of the Dynex SDK account:

    ACCOUNT: <YOUR ACCOUNT IDENTIFICATION>
    API SUCCESSFULLY CONNECTED TO DYNEX
    -----------------------------------
    ACCOUNT LIMITS:
    MAXIMUM NUM_READS: 5,000,000
    MAXIMUM ANNEALING_TIME: 10,000
    MAXIMUM JOB DURATION: 60 MINUTES
    
    USAGE:
    TOTAL USAGE: 97,699,614,400 / 10,000,000,000,000 ( 0.976996144 %) NUM_READS x ANNEALING_TIME

    """
    
    check_api_status(logging = True);

def check_api_status(logging = False):
    """
    Internal Function
    -----------------

    Dynex API call to output the status of the Dynex SDK account

    Returns:
    --------
    TRUE if the API call was successful
    FALSE if the API call was not successful
    """
    
    global FILE_IV
    global FILE_KEY
    global MAX_CHIPS, MAX_ANNEALING_TIME, MAX_DURATION, TOTAL_USAGE
    url = API_ENDPOINT+'?api_key='+API_KEY+'&api_secret='+API_SECRET+'&method=status'
    with urllib.request.urlopen(url) as ret:
        data = json.load(ret);
        error = data['error'];
        status = data['status'];
    retval = False;
    if error == False and status == 'valid':
        FILE_IV = str.encode(data['i']);
        FILE_KEY = data['k'];
        MAX_CHIPS = data['max_chips'];
        MAX_ANNEALING_TIME = data['max_steps'];
        MAX_DURATION = data['max_duration'];
        MAX_USAGE = data['max_usage'];
        TOTAL_USAGE = data['total_usage'];
        ACCOUNT_NAME = data['account_name'];
        if logging:
            print('ACCOUNT:',ACCOUNT_NAME);
            print('API SUCCESSFULLY CONNECTED TO DYNEX');
            print('-----------------------------------');
            print('ACCOUNT LIMITS:');
            print('MAXIMUM NUM_READS:','{:,}'.format(MAX_CHIPS));
            print('MAXIMUM ANNEALING_TIME:','{:,}'.format(MAX_ANNEALING_TIME));
            print('MAXIMUM JOB DURATION:','{:,}'.format(MAX_DURATION),'MINUTES')
            print('');
            print('USAGE:');
            usage_pct = TOTAL_USAGE / MAX_USAGE * 100.0;
            print('TOTAL USAGE:','{:,}'.format(TOTAL_USAGE),'/','{:,}'.format(MAX_USAGE),'(',usage_pct,'%)','NUM_READS x ANNEALING_TIME');
        retval = True;
    else:
        raise Exception('INVALID API CREDENTIALS');
    return retval;

def update_job_api(JOB_ID, status, logging=True, workers=-1, lowest_loc=-1, lowest_energy=-1):
    """
    Internal Function
    -----------------

    Dynex API call to update the status of a job

    Returns:
    --------
    TRUE if the job was successfully updated
    FALSE if there was a problem with updating the job
    """
    
    url = API_ENDPOINT+'?api_key='+API_KEY+'&api_secret='+API_SECRET+'&method=update_job&job_id='+str(JOB_ID)+'&status='+str(status);
    url += '&workers='+str(workers)+'&lowest_loc='+str(lowest_loc)+'&lowest_energy='+str(lowest_energy);
    with urllib.request.urlopen(url) as ret:
        data = json.load(ret);
        error = data['error'];
    retval = False;
    if error == False:
        retval = True;
        if logging:
            print("[DYNEX] MALLOB: JOB UPDATED:",JOB_ID,"STATUS:",status);
    else:
        print("[DYNEX] ERROR DURING UPDATING JOB ON MALLOB");
        raise Exception('ERROR DURING UPDATING JOB ON MALLOB');
    return retval;

def generate_job_api(sampler, annealing_time, switchfraction, num_reads, alpha=20, beta=20, gamma=1, delta=1, epsilon=1, zeta=1, minimum_stepsize=0.00000006, logging=True):
    """
    Internal Function
    -----------------

    Dynex API call to generate a new job

    Returns:
    --------
    TRUE if the job was successfully created
    FALSE if there was a problem with generating the job
    """
    
	# retrieve additional data from sampler class:
    sampler_type = sampler.type;
    sampler_num_clauses = sampler.num_clauses;
    filehash = sampler.filehash;
    description = base64.b64encode(sampler.description.encode('ascii')).decode('ascii');
    filename = base64.b64encode(sampler.filename.encode('ascii')).decode('ascii');
    downloadurl = base64.b64encode(sampler.downloadurl.encode('ascii')).decode('ascii');
    solutionurl = base64.b64encode(sampler.solutionurl.encode('ascii')).decode('ascii');
    solutionuser = base64.b64encode(sampler.solutionuser.encode('ascii')).decode('ascii');

    url = API_ENDPOINT+'?api_key='+API_KEY+'&api_secret='+API_SECRET+'&method=generate_job&annealing_time='+str(annealing_time)+'&switchfraction='+str(switchfraction);
    url += '&num_reads='+str(num_reads)+'&alpha='+str(alpha)+'&beta='+str(beta)+'&gamma='+str(gamma)+'&delta='+str(delta)+'&epsilon='+str(epsilon)+'&zeta='+str(zeta);
    url += '&minimum_stepsize='+str(minimum_stepsize)+'&sampler_type='+sampler_type+'&num_clauses='+str(sampler_num_clauses)
    url += '&filehash='+filehash+'&description='+description+'&filename='+filename+'&downloadurl='+downloadurl+'&solutionurl='+solutionurl+'&solutionuser='+solutionuser;
    with urllib.request.urlopen(url) as ret:
        data = json.load(ret);
        error = data['error'];
    retval = False;
    if error == False:
        retval = True;
        if logging:
            print("[DYNEX] MALLOB: JOB CREATED: ",data['job_id']);
        return int(data['job_id']);
    else:
        print("[DYNEX] ERROR CREATING JOB:",data['message']);
        raise Exception(data['message']);
    return retval;

def get_status_details_api(JOB_ID, all_stopped = False):
    """
    Internal Function
    -----------------

    Dynex API call to retrieve status of the job

    Returns:
    --------
    LOC_MIN:
    Lowest value of global falsified soft clauses of the problem which is being sampled.
    
    ENERGY_MIN:
    Lowest QUBO energy of the problem which is being sampled
    
    CHIPS:
    The number of chips which are currently sampling
    
    retval:
    Tabulated overview of the job status, showing workers, found assignments, etc.
    """

    url = API_ENDPOINT+'?api_key='+API_KEY+'&api_secret='+API_SECRET+'&method=get_status&job_id='+str(JOB_ID)
    with urllib.request.urlopen(url) as ret:
            data = json.load(ret);
    table = [['WORKER','VERSION','CHIPS','LOC','ENERGY','RUNTIME','LAST UPDATE', 'STATUS']];
    LOC_MIN = 1.7976931348623158e+308;
    ENERGY_MIN = 1.7976931348623158e+308;
    CHIPS = 0;
    i = 0;
    for result in data:
        worker = result['worker_id'];
        chips = result['chips'];
        started = result['created_at'];
        updated = result['updated_at'];
        loc = result['loc'];
        energy = "{:.2f}".format(result['energy']);
        interval = "{:.2f}".format(result['lastupdate']/60)+' min';
        version = result['version'];
        lastupdate = "{:.2f}s ago".format(result['runtime'])

        status = "\033[1;31m%s\033[0m" %'STOPPED';
        if result['runtime']<=60:
            status = "\033[1;32m%s\033[0m" %'RUNNING';
        if all_stopped:
            status = "\033[1;31m%s\033[0m" %'STOPPED';

        table.append([worker, version, chips, loc, energy, interval, lastupdate, status]);

        CHIPS = CHIPS + result['chips'];
        if result['loc'] < LOC_MIN:
            LOC_MIN = result['loc'];
        if result['energy'] < ENERGY_MIN:
            ENERGY_MIN = result['energy'];
        i = i + 1;
    # not worked on:
    if i==0:
        table.append(['*** WAITING FOR WORKERS ***','','','','','','','']);
        LOC_MIN = 0;
        ENERGY_MIN = 0;
        CHIPS = 0;

    retval = tabulate(table, headers="firstrow", tablefmt="rounded_grid", stralign="right", floatfmt=".2f")

    return LOC_MIN, ENERGY_MIN, CHIPS, retval;

################################################################################################################################
# TEST FTP ACCESS
################################################################################################################################

def test_completed():
    """
    Internal Function
    -----------------

    Returns TRUE if dynex.test() has been successfully completed
    Returns FALSE if dynex.test() was not successfully completed
    """
    
    local_path='dynex.test';
    return os.path.isfile(local_path);

def test():
    """
    Internal Function
    -----------------

    Performs test of the dynex.ini settings. Successful completion is required to start using the sampler.
    """

    allpassed = True;
    print('[DYNEX] TEST: dimod BQM construction...')
    bqm = dimod.BinaryQuadraticModel({'a': 1.5}, {('a', 'b'): -1}, 0.0, 'BINARY')
    model = BQM(bqm, logging=False);
    print('[DYNEX] PASSED');
    print('[DYNEX] TEST: Dynex Sampler object...')
    sampler = DynexSampler(model,  mainnet=False, logging=False, test=True);
    print('[DYNEX] PASSED');
    print('[DYNEX] TEST: uploading computing file...')
    ret = upload_file_to_ftp(sampler.ftp_hostname, sampler.ftp_username, sampler.ftp_password, sampler.filepath+sampler.filename, sampler.ftp_path, sampler.logging);
    if ret==False:
        allpassed=False;
        print('[DYNEX] FAILED');
        raise Exception("DYNEX TEST FAILED");
    else:
        print('[DYNEX] PASSED');
    time.sleep(1)
    print('[DYNEX] TEST: submitting sample file...')
    worker_user = sampler.solutionuser.split(':')[0]
    worker_pass = sampler.solutionuser.split(':')[1]
    ret = upload_file_to_ftp(sampler.solutionurl[6:-1], worker_user, worker_pass, sampler.filepath+sampler.filename, '', sampler.logging);
    if ret==False:
        allpassed=False;
        print('[DYNEX] FAILED');
        raise Exception("DYNEX TEST FAILED");
    else:
        print('[DYNEX] PASSED');
    time.sleep(1)
    print('[DYNEX] TEST: retrieving samples...')
    try:
        files = list_files_with_text(sampler);
        print('[DYNEX] PASSED');
    except:
        allpassed=False;
        print('[DYNEX] FAILED');
        raise Exception("DYNEX TEST FAILED");

    time.sleep(1)
    print('[DYNEX] TEST: worker access to computing files')
    url = sampler.downloadurl + sampler.filename
    try:
        with urllib.request.urlopen(url) as f:
            html = f.read().decode('utf-8');
            print('[DYNEX] PASSED');
    except:
        allpassed=False;
        print('[DYNEX] FAILED');
        raise Exception("DYNEX TEST FAILED");

    if allpassed:
        print('[DYNEX] TEST RESULT: ALL TESTS PASSED');
        with open('dynex.test', 'w') as f:
            f.write('[DYNEX] TEST RESULT: ALL TESTS PASSED')
    else:
        print('[DYNEX] TEST RESULT: ERRORS OCCURED');

################################################################################################################################
# conversation of k-sat to 3sat
################################################################################################################################

def check_list_length(lst):
    """
    Internal Function
    -----------------

    Returns TRUE if the sat problem is k-Sat 
    Returns FALSE if the problem is 3-sat or 2-sat
    """
    
    for sublist in lst:
        if isinstance(sublist, list) and len(sublist) > 3:
            return True
    return False

# find largest variable in clauses:
def find_largest_value(lst):
    """
    Internal Function
    -----------------

    Returns the largest variable in a list of clauses.
    """
    
    largest_value = None

    for sublist in lst:
        for value in sublist:
            if largest_value is None or value > largest_value:
                largest_value = value

    return largest_value

# create a substitution clause:
def sat_creator(variables, clause_type, dummy_number, results_clauses):
    """
    Internal Function
    -----------------

    Converts a k-sat clause to a number of 3-sat clauses.
    """
    
    if clause_type == 1:
        #Beginning clause
        results_clauses.append([variables[0], variables[1], dummy_number])
        dummy_number *= -1

    elif clause_type == 2:
        #Middle clause
        for i in range(len(variables)):
            temp = dummy_number
            dummy_number *= -1
            dummy_number += 1
            results_clauses.append([temp, variables[i], dummy_number])
            dummy_number *= -1

    elif clause_type == 3:
        #Final clause
        for i in range(len(variables)-2):
            temp = dummy_number
            dummy_number *= -1
            dummy_number += 1
            results_clauses.append([temp, variables[i], dummy_number])
            dummy_number *= -1   
        results_clauses.append([dummy_number, variables[-2], variables[-1]])
        dummy_number *= -1
        dummy_number += 1
        
    return dummy_number, results_clauses

# convert from k-sat to 3sat:
def ksat(clauses):
    """
    Internal Function
    -----------------

    Converts a k-sat formulation into 3-sat.

    Returns:
    --------
    List of clauses of the converted 3-sat
    """
    
    results_clauses = [];
    results_clauses.append([1])
    variables = find_largest_value(clauses);
    dummy_number = variables + 1;
    for values in clauses:
        total_variables = len(values)
        #Case 3 variables
        if total_variables == 3:
            results_clauses.append([values[0], values[1], values[2]])
        else:
            #Case 1 variable
            if total_variables == 1:
                results_clauses.append([values[0]])
            #Case 2 variables
            elif total_variables == 2:
                results_clauses.append([values[0], values[1]])
                dummy_number += 1
            #Case more than 3 variable
            else:
                first_clause = values[:2]
                dummy_number, results_clauses = sat_creator(first_clause, 1, dummy_number, results_clauses)

                middle_clauses = values[2:len(values)-2]
                dummy_number, results_clauses = sat_creator(middle_clauses, 2, dummy_number, results_clauses)

                last_clause = values[len(values)-2:]
                dummy_number, results_clauses = sat_creator(last_clause, 3, dummy_number, results_clauses)

    return results_clauses

################################################################################################################################
# utility functions
################################################################################################################################

def calculate_sha3_256_hash(string):
    sha3_256_hash = hashlib.sha3_256()
    sha3_256_hash.update(string.encode('utf-8'))
    return sha3_256_hash.hexdigest()

def Convert(a):
    it = iter(a)
    res_dct = dict(zip(it, it))
    return res_dct;

def check_list_length(lst):
    for sublist in lst:
        if isinstance(sublist, list) and len(sublist) > 3:
            return True
    return False

def max_value(inputlist):
    return max([sublist[-1] for sublist in inputlist])

################################################################################################################################
# upload file to an FTP server
################################################################################################################################

def upload_file_to_ftp(hostname, username, password, local_file_path, remote_directory, logging=True):
    """
    Internal Function
    -----------------

    Submits a computation file (xxx.bin) to the FTP server as defined in dynex.ini
    """
    
    retval = True;
    try:
        ftp = FTP(hostname)
        ftp.login(username, password)
        # Change to the remote directory
        ftp.cwd(remote_directory)
    
        # Open the local file in binary mode for reading
        with open(local_file_path, 'rb') as file:
            total = os.path.getsize(local_file_path); # file size
            # sanity check:
            if total > 104857600:
                print("[ERROR] PROBLEM FILE TOO LARGE (MAX 104,857,600 BYTES)");
                raise Exception('PROBLEM FILE TOO LARGE (MAX 104,857,600 BYTES)');
    
            if logging:
                with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024, desc='file upload progress') as pbar:
                    
                    def cb(data):
                        pbar.update(len(data))
                    
                    # Upload the file to the FTP server
                    ftp.storbinary(f'STOR {local_file_path.split("/")[-1]}', file, 1024, cb)
            else:
                # Upload the file to the FTP server
                ftp.storbinary(f'STOR {local_file_path.split("/")[-1]}', file)
    
        if logging:
            print(f"[DYNEX] File '{local_file_path}' sent successfully to '{hostname}/{remote_directory}'")

    except Exception as e:
        print(f"[DYNEX] An error occurred while sending the file: {str(e)}")
        retval = False;
    finally:
        ftp.quit();
    return retval;

################################################################################################################################
# Cleanup FTP on sampler exit or clean()
################################################################################################################################
def cleanup_ftp(sampler, files):
    """
    Internal Function
    -----------------

    This function is called on __exit__ of the sampler class or by sampler.clear(). 
    It ensures that submitted sample-files,
    which have not been parsed and used from the sampler, will be deleted on the FTP server. 
    """
    
    if len(files)>0:
        try:
            host = sampler.solutionurl[6:-1];
            username = sampler.solutionuser.split(":")[0];
            password = sampler.solutionuser.split(":")[1]; 
            directory = "";
            ftp = FTP(host);
            ftp.login(username, password);
            ftp.cwd(directory);
            for file in files:
                ftp.delete(file);
            if sampler.logging:
                print("[ÐYNEX] FTP DATA CLEANED");
        except Exception as e:
            print(f"[DYNEX] An error occurred while deleting file: {str(e)}")
        finally:
            ftp.quit();
    return;

################################################################################################################################
# delete computing file on an FTP server
################################################################################################################################
def delete_file_on_ftp(hostname, username, password, local_file_path, remote_directory, logging=True):
    """
    Internal Function
    -----------------

    Deletes a file on the FTP server as specified in dynex,ini
    """
    
    ftp = FTP(hostname)
    ftp.login(username, password)
    # Change to the remote directory
    ftp.cwd(remote_directory)
    ftp.delete(local_file_path.split("/")[-1]);
    if logging:
        print("[DYNEX] COMPUTING FILE", local_file_path.split("/")[-1],'REMOVED');
    return

################################################################################################################################
# retrieve all files starting with "sampler.filename" from an FTP server
################################################################################################################################

def list_files_with_text(sampler):
    """
    Internal Function
    -----------------

    Downloads assignment files from the FTP server specified in dynex.ini and stores them in /tmp as specified in dynex.ini
    Downloaded files are automatically deleted on the FTP server.

    Returns:
    --------
    List of locally in /tmp saved assignment files for the current sampler model.
    """
    
    host = sampler.solutionurl[6:-1];
    username = sampler.solutionuser.split(":")[0];
    password = sampler.solutionuser.split(":")[1]; 
    directory = "";
    text = sampler.filename;
    # Connect to the FTP server
    ftp = FTP(host)
    ftp.login(username, password)
    
    # Change to the specified directory
    ftp.cwd(directory)
    
    # List all (fully uploaded) files in the directory (minimum size)
    target_size = 97 + sampler.num_variables;
    filtered_files = [];
    for name, facts in ftp.mlsd(): 
        if 'size' in facts:
            if int(facts['size'])>=target_size and name.startswith(text):
                filtered_files.append(name);
                # download file if not already local:
                local_path = sampler.filepath+name;
                if os.path.isfile(local_path)==False:	
                    with open(local_path, 'wb') as file:
                        ftp.retrbinary('RETR ' + name, file.write); 
                        file.close();
                        # we delete downloaded files from FTP:
                        ftp.delete(name); 
    
    # Close the FTP connection
    ftp.quit()

    # In our status view, we show the local, downloaded and available files:
    filtered_files = list_files_with_text_local(sampler);
    
    return filtered_files

################################################################################################################################
# retrieve all files starting with "sampler.filename" from test-net
################################################################################################################################

def list_files_with_text_local(sampler):
    """
    Internal Function
    -----------------

    Returns a list of all assignment files (filenames) which are locally available in /tmp as specified in dynex.ini for
    the current sampler model.
    """
    
    directory = sampler.filepath_full; 
    fn = sampler.filename+".";
    # list to store files
    filtered_files = []

    for filename in os.listdir(directory):
        if filename.startswith(fn):
            filtered_files.append(filename)

    return filtered_files;    
    
################################################################################################################################
# Download file from FTP to sampler.filepath / filename
################################################################################################################################

def download_file(sampler, filename):
    """
    Internal Function
    -----------------

    Downloads a computed assigment file from the FTP server specified in dynex.ini 
    """
    
    host = sampler.solutionurl[6:-1];
    username = sampler.solutionuser.split(":")[0];
    password = sampler.solutionuser.split(":")[1]; 
    directory = "";
    local_path = sampler.filepath+filename;
    # Connect to the FTP server
    ftp = FTP(host)
    ftp.login(username, password)
    
    # Change to the specified directory
    ftp.cwd(directory)
    
    # Download the file
    with open(local_path, 'wb') as file:
        ftp.retrbinary('RETR ' + filename, file.write); # download file locally
        ftp.delete(filename); # remove file from FTP
    
    # Close the FTP connection
    ftp.quit()

################################################################################################################################
# generate filehash for worker
################################################################################################################################

def generate_hash(filename):
    """
    Internal Function
    -----------------

    Returns the sha3-256 hash of a given file. 
    """
    
    with open(filename, 'r') as file:
        data = file.read().replace('\n', '');
    return calculate_sha3_256_hash(data);

################################################################################################################################
# AES Encryption / Decryption class
################################################################################################################################

def aes_encrypt(raw):
    """
    Internal Function
    -----------------

    Returns the encrypted string of 'raw' with an AES Key privided by the Dynex platform. 
    """
    BLOCK_SIZE = 16
    pad = lambda s: s + (BLOCK_SIZE - len(s) % BLOCK_SIZE) * chr(BLOCK_SIZE - len(s) % BLOCK_SIZE)
    unpad = lambda s: s[:-ord(s[len(s) - 1:])]
    raw = pad(raw);
    cipher = AES.new(FILE_KEY.encode("utf8"), AES.MODE_CBC, FILE_IV);
    output = cipher.encrypt(raw.encode("utf8"));
    output_str = binascii.hexlify(output);
    output_str = str(output_str)[2:-1];
    return output_str

################################################################################################################################
# save clauses to SAT cnf file
################################################################################################################################

def save_cnf(clauses, filename, mainnet):
    """
    Internal Function
    -----------------

    Saves the model as an encrypted .bin file locally in /tmp as defined in dynex.ini 
    """
    
    num_variables = max(max(abs(lit) for lit in clause) for clause in clauses);
    num_clauses = len(clauses);
    
    with open(filename, 'w') as f:
        line = "p cnf %d %d" % (num_variables, num_clauses);
        
        if mainnet:
            line_enc = aes_encrypt(line);
        else:
            line_enc = line;
        
        f.write(line_enc+"\n"); 
        
        for clause in clauses:
            line = ' '.join(str(int(lit)) for lit in clause) + ' 0';
        
            if mainnet:
                line_enc = aes_encrypt(line);
            else:
                line_enc = line;
        
            f.write(line_enc+"\n");

################################################################################################################################
# save wcnf file
################################################################################################################################

def save_wcnf(clauses, filename, num_variables, num_clauses, mainnet):
    """
    Internal Function
    -----------------

    Saves the model as an encrypted .bin file locally in /tmp as defined in dynex.ini 
    """

    with open(filename, 'w') as f:
        line = "p wcnf %d %d" % (num_variables, num_clauses);
        
        if mainnet:
            line_enc = aes_encrypt(line);
        else:
            line_enc = line;

        f.write(line_enc+"\n"); 

        for clause in clauses:
            line = ' '.join(str(int(lit)) for lit in clause) + ' 0';
        
            if mainnet:
                line_enc = aes_encrypt(line);
            else:
                line_enc = line;
        
            f.write(line_enc+"\n"); 

        
################################################################################################################################
# functions to convert BQM to wcnf
################################################################################################################################

def convert_bqm_to_wcnf(bqm, relabel=True, logging=True):
    """
    Internal Function
    -----------------

    Converts a given Binary Quadratic Model (BQM) problem into a wncf file which is being used by the Dynex platform workers
    for the sampling process. Every BQM can be converted to a QUBO formulation in polynomial time (and vice-versa) without
    loss of functionality. During the process, variables are re-labeld and mapped to integer values in the range of 
    [0, NUM_VARIABLES}. The mapping is being made available in sampler.variable_mappings and is used for constructing the
    returned sampleset object.

    Note: 
    * The BQM needs to have at least one defined weight, otherwise an exception is thrown
    * Double values of weights are being converted to integer values with the factor 'PRECISION' 
    * The value for PRECISION is determined automatically with function 10 ** (np.floor(np.log10(max_abs_coeff)) - 4)

    Returns:
    --------
    clauses:
    A list of all clauses
    
    num_variables:
    Integer: number of variables
    
    num_clauses:
    Integer: number of clauses
    
    mappings:
    Dictionary: variable mappings original -> integer value
    
    precision:
    Double: precision of conversion
    
    bqm:
    BQM object
    """
    
    # relabel variables to integers:
    mappings = bqm.variables._relabel_as_integers();
    
    # convert bqm to_qubo model:
    clauses = [];
    Q = bqm.to_qubo();
    Q_list = list(Q[0]);
    if logging:
        print("[DYNEX] MODEL CONVERTED TO QUBO")
    
    # precision:
    newQ = [];
    for i in range(0, len(Q_list)):
        touple = Q_list[i];
        w = Q[0][touple];
        newQ.append(w);
    max_abs_coeff = np.max(np.abs(newQ));
    if max_abs_coeff == 0:
        print('[DYNEX] ERROR: AT LEAST ONE WEIGHT MUST BE > 0.0');
        raise Exception("ERROR: AT LEAST ONE WEIGHT MUST BE > 0.0");

    precision = 10 ** (np.floor(np.log10(max_abs_coeff)) - 4);

    # max precision is 1:
    if precision>1:
        if logging:
            print("[ÐYNEX] PRECISION CUT FROM",precision,"TO 1");
        precision = 1;

    if logging:
        print("[DYNEX] PRECISION SET TO", precision);

    # constant offset:
    W_add = Q[1]; 
    if logging:
        print("[DYNEX] QUBO: Constant offset of the binary quadratic model:", W_add);

    for i in range(0, len(Q_list)):
        touple = Q_list[i];
        i = int(touple[0])+1; # +1 because vars need to start with 1
        j = int(touple[1])+1; # +1 because vars need to start with 1
        w = Q[0][touple];
        w_int = int(np.round(w/precision));
        
        # linear term:
        if i==j:
            if w_int > 0:
                clauses.append([w_int,-i]);
            if w_int < 0:
                clauses.append([-w_int, i]);
        
        # quadratic term:
        if i!=j:
            if w_int > 0:
                clauses.append([w_int, -i, -j]);
            if w_int < 0:
                clauses.append([-w_int, i, -j]);
                clauses.append([-w_int, j]);
        
    num_variables = len(bqm.variables);
    num_clauses = len(clauses);

    # re-map variables from integer to original using self.var_mapping:
    bqm.variables._relabel(mappings)
    
    return clauses, num_variables, num_clauses, mappings, precision, bqm

################################################################################################################################
# Supported Model Classes
################################################################################################################################

class SAT():
    """
    Creates a model, which is being used by the sampler based on a SAT problem.
    """
    def __init__(self, clauses, logging=True):
        self.clauses = clauses;
        self.type = 'cnf';
        self.bqm = "";
        self.logging = logging;
        self.typestr = 'SAT';

class BQM():
    """
    Creates a model, which is being used by the sampler based on a Binary Quadratic Model (BQM) problem.
    """
    def __init__(self, bqm, relabel=True, logging=True):
        self.clauses, self.num_variables, self.num_clauses, self.var_mappings, self.precision, self.bqm = convert_bqm_to_wcnf(bqm, relabel, logging);
        if self.num_clauses == 0 or self.num_variables == 0:
            return;
        self.type = 'wcnf';
        self.logging = logging;
        self.typestr = 'BQM';

class CQM():
    """
    Creates a model, which is being used by the sampler based on a Constraint Quadratic Model (CQM) problem.
    """
    def __init__(self, cqm, relabel=True, logging=True):
        bqm, self.invert = dimod.cqm_to_bqm(cqm)
        self.clauses, self.num_variables, self.num_clauses, self.var_mappings, self.precision, self.bqm = convert_bqm_to_wcnf(bqm, relabel, logging);
        self.type = 'wcnf';
        self.logging = logging;
        self.typestr = 'CQM';

################################################################################################################################
# Dynex Sampler class
################################################################################################################################

class DynexSampler:
    def __init__(self, model, logging=True, mainnet=True, description='Dynex SDK Job', test=False):
        """
        Initialises the sampler object given a model.

        logging:
        [TRUE, FALSE]
        Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (TRUE)

        mainnet:
        [TRUE, FALSE]
        Defines if the mainnet (Dynex platform sampling) or the testnet (local sampling) is being used for sampling

        description:
        STRING
        Defines the description for the sampling, which is shown in Dynex job dashboards as well as in the market place.

        test:
        For internal use only.
        
        """

        if not test and not test_completed():
            raise Exception("CONFIGURATION TEST NOT COMPLETED. PLEASE RUN 'dynex.test()'");
        
        self.description = description;

        # parse config file:
        config = configparser.ConfigParser();
        config.read('dynex.ini', encoding='UTF-8');
        
        # SDK Authenticaton:
        if not check_api_status():
            raise Exception("API credentials invalid");

        # FTP & HTTP GET data where miners are accessing problem files:
        self.ftp_hostname = config['FTP_COMPUTING_FILES']['ftp_hostname'];
        self.ftp_username = config['FTP_COMPUTING_FILES']['ftp_username'];
        self.ftp_password = config['FTP_COMPUTING_FILES']['ftp_password'];
        self.ftp_path     = config['FTP_COMPUTING_FILES']['ftp_path'];
        self.downloadurl  = config['FTP_COMPUTING_FILES']['downloadurl'];
        
        # FTP data where miners submit results:
        self.solutionurl  = 'ftp://'+config['FTP_SOLUTION_FILES']['ftp_hostname']+'/';
        self.solutionuser = config['FTP_SOLUTION_FILES']['ftp_username']+":"+config['FTP_SOLUTION_FILES']['ftp_password'];
        
        # local path where tmp files are stored
        tmppath = Path("tmp/test.bin");
        tmppath.parent.mkdir(exist_ok=True);
        with open(tmppath, 'w') as f:
            f.write('0123456789ABCDEF')
        self.filepath = 'tmp/'
        self.filepath_full = os.getcwd()+'/tmp'

        # path to testnet
        self.solverpath = 'testnet/';

        # auto generated temp filename:
        self.filename = secrets.token_hex(16)+".bin";
        self.logging = logging;
        self.mainnet = mainnet;
        self.typestr = model.typestr;
        
        if model.type == 'cnf':
            # convert to 3sat?
            if (check_list_length(model.clauses)):
                # we need to convert to 3sat:
                self.clauses = ksat(model.clauses);
            else:
                self.clauses = model.clauses;
            save_cnf(self.clauses, self.filepath+self.filename, mainnet);
            self.num_clauses = len(self.clauses);
            self.num_variables = max_value(self.clauses) - 1;
        
        if model.type == 'wcnf':
            self.clauses = model.clauses;
            self.num_variables = model.num_variables;
            self.num_clauses = model.num_clauses;
            self.var_mappings = model.var_mappings;
            self.precision = model.precision;
            save_wcnf(self.clauses, self.filepath+self.filename, self.num_variables, self.num_clauses, mainnet); 

        self.filehash     = generate_hash(self.filepath+self.filename);
        self.type = model.type;
        self.assignments = {};
        self.dimod_assignments = {};
        self.bqm = model.bqm;

        if self.logging:
            print("[DYNEX] SAMPLER INITIALISED")

    def clean(self):
        """
        This function should be called after finishing a sampling process on the Mainnet. It ensures that submitted sample-files,
        which have not been parsed and used from the sampler, will be deleted on the FTP server. It is also called automatically 
        during __exit___ event of the sampler class.
        """
        if self.mainnet:
            files = list_files_with_text(self); 
            cleanup_ftp(self, files);

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Upon __exit__, the function clean() is being called.
        """
    	# delete remaining, not parsed files on FTP server:
        self.clean();
        print('[DYNEX] SAMPLER EXIT');
        
    def update(self, model, logging=True):
        """
        Typically, the sampler object is being initialised with a defined model class. This model can also be updated without
        regenerating a new sampler object by calling the function update(model).
        """
        self.logging = logging;
        self.filename     = secrets.token_hex(16)+".bin"; 
        
        if model.type == 'cnf':
            # convert to 3sat?
            if (check_list_length(model.clauses)):
                self.clauses = ksat(model.clauses);
            else:
                self.clauses = model.clauses;
            save_cnf(self.clauses, self.filepath+self.filename);
        
        if model.type == 'wcnf':
            self.clauses = model.clauses;
            self.num_variables = model.num_variables;
            self.num_clauses = model.num_clauses;
            self.var_mappings = model.var_mappings;
            self.precision = model.precision;
            save_wcnf(self.clauses, self.filepath+self.filename, self.num_variables, self.num_clauses, self.mainnet); 
        
        self.filehash     = generate_hash(self.filepath+self.filename);
        self.type = model.type;
        self.assignments = {};
        self.dimod_assignments = {};
        self.bqm = model.bqm;

    # print summary of sampler:
    def print(self):
        """
        Prints summary information about the sampler object:

        * Mainnet: [TRUE, FALSE] If the mainnet (Dynex platform sampling) or the testnet (local sampling) is being used for sampling
        * logging: [TRUE, FALSE] Show progress and status information or be quiet (not terminal output)
        * tmp filename: The filename of the computation file
        * tmp_filehash: The checksum hash of the computation file
        * model type: [cnf, wcnf]: The type of the model: Sat problems (cnf) or QUBU/Ising type problems (wcnf)
        * num_variables: The number of variables of the model
        * num_clauses: The number of clauses of the model
        """
        print('{DynexSampler object}');
        print('mainnet?', self.mainnet);
        print('logging?', self.logging);
        print('tmp filename:',self.filepath+self.filename);
        print('tmp filehash:',self.filehash);
        print('model type:', self.typestr);
        print('num variables:', self.num_variables);
        print('num clauses:', self.num_clauses);
        print('configuration: dynex.ini');

    # convert a sampler.sampleset[x]['sample'] into an assignment:
    def sample_to_assignments(self, lowest_set):
        """
        The voltates of a sampling can be retrieved from the sampler with sampler.sampleset

        The sampler.sampleset returns a list of voltages for each variable, ranging from -1.0 to +1.0 and is a double precision value.
        Sometimes it is required to transform these voltages to binary values 0 (for negative voltages) or 1 (for positive voltages).
        This function converts a given sampler.sampleset[x] from voltages to binary values.

        Returns:
        --------
        Returns the converted sample as list
        """
        sample = {};
        i = 0;
        for var in self.var_mappings:
            sample[var] = 1;
            if (float(lowest_set[i])<0):
                sample[var] = 0;
            i = i + 1
        return sample;
        
    # sampling process:
    def sample(self, num_reads = 32, annealing_time = 10, switchfraction = 0.0, alpha=20, beta=20, gamma=1, delta=1, epsilon=1, zeta=1, minimum_stepsize = 0.00000006, debugging=False):
        """
        The main sampling function:

        num_reads: 
        Integer value in the range of [32, MAX_NUM_READS] as defined in your license.
        Defines the number of parallel samples to be performed.

        annealing_time:
        Integer value in the range of [1, MAX_ANNEALING_TIME] as defined in your license.
        Defines the number of integration steps for the sampling. Models are being converted into neuromorphic circuits, which are then
        simulated with ODE integration by the participating workers.

        switchfraction:
        Double value in the range of [0.0, 1.0] 
        Defines the percentage of variables which are replaced by random values during warm start samplings. 

        alpha, beta, gamma, delta, epsilon, zeta
        Double values in the range of of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon.
        The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters
        for the ODE integration. Setting values defines the upper bound for the automated parameter tuning.

        minimum_stepsize:
        Double value in the range of [0.0000000000000001, 1.0]
        The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is performig adaptive stepsizes for each
        ODE integration forward Euler step. This value defines the smallest step size for a single adaptive integration step.

        debugging:
        [TRUE, FALSE]
        Only applicable for test-net sampling. Defines if the sampling process should be quiet with no terminal output (FALSE) 
        or if process updates are to be shown (TRUE)

        Returns:
        --------
        Returns a dimod sampleset object
        """
        
        mainnet = self.mainnet;

        try:
        
            # step 1: upload problem file to Dynex Platform: ---------------------------------------------------------------------------------
            if mainnet:
                # create job on mallob system:
                JOB_ID = generate_job_api(self, annealing_time, switchfraction, num_reads, alpha, beta, gamma, delta, epsilon, zeta, minimum_stepsize, self.logging);
                # upload job:
                if self.logging:
                    print("[ÐYNEX] SUBMITTING JOB - UPLOADING JOB FILE...");
                ret = upload_file_to_ftp(self.ftp_hostname, self.ftp_username, self.ftp_password, self.filepath+self.filename, self.ftp_path, self.logging);
                if ret == False:
                    raise Exception("[DYNEX] ERROR: FILE UPLOAD FAILED.");

                # now set the job as ready to be worked on:
                if self.logging:
                    print("[ÐYNEX] SUBMITTING START COMMAND...");
                if not update_job_api(JOB_ID, 0, self.logging):
                    raise Exception('ERROR: CANNOT START JOB')
                
                if self.logging:
                    print("[ÐYNEX] STARTING JOB...");
            else:
                # run on test-net:
                if self.type == 'wcnf':
                    localtype = 5;
                if self.type == 'cnf':
                    localtype = 0;
                JOB_ID = -1;
                command = self.solverpath+"np -t="+str(localtype)+" -ms="+str(annealing_time)+" -st=1 -msz="+str(minimum_stepsize)+" -c="+str(num_reads)+" --file='"+self.filepath_full+"/"+self.filename+"'";

                if alpha!=0:
                    command = command + " --alpha=" + str(alpha);
                if beta!=0:
                    command = command + " --beta=" + str(beta);
                if gamma!=0:
                    command = command + " --gamma=" + str(gamma);
                if delta!=0:
                    command = command + " --delta=" + str(delta);
                if epsilon!=0:
                    command = command + " --epsilon=" + str(epsilon);
                if zeta!=0:
                    command = command + " --zeta=" + str(zeta);

                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                if debugging:
                    for c in iter(lambda: process.stdout.read(1), b""):
                        sys.stdout.write(c.decode('utf-8'))
                else:
                    if self.logging:
                    	print("[DYNEX|TESTNET] *** WAITING FOR READS ***");
                    process.wait();

            # step 2: wait for process to be finished: -------------------------------------------------------------------------------------
            t = time.process_time();
            finished = False;
            runupdated = False;
            cnt_workers = 0;

            while finished==False:
                total_chips = 0;
                total_steps = 0;
                lowest_energy = 1.7976931348623158e+308;
                lowest_loc = 1.7976931348623158e+308;

                # retrieve solutions
                if mainnet:
                    try:
                        files = list_files_with_text(self);
                        cnt_workers = len(files);
                    except Exception as e:
                        print('[DYNEX] CONNECTION TO FTP ENDPOINT FAILED:',e);
                        raise Exception('ERROR: ONNECTION TO FTP ENDPOINT FAILED')
                        files = []; 
                else:
                    files = list_files_with_text_local(self); 
                    time.sleep(1);

                for file in files:
                    info = file[len(self.filename)+1:];
                    chips = int(info.split(".")[0]);
                    steps = int(info.split(".")[1]);
                    loc = int(info.split(".")[2]);
                    # energy can also be non decimal:
                    if len(info.split("."))>4:
                        energy = float(info.split(".")[3]+"."+info.split(".")[4]);
                    else:
                        energy = float(info.split(".")[3]);
                    total_chips = total_chips + chips;
                    total_steps = steps;
                    if energy < lowest_energy:
                        lowest_energy = energy;
                    if loc < lowest_loc:
                        lowest_loc = loc;
                    if self.type=='cnf' and loc == 0:
                        finished = True;
                    if total_chips >= num_reads*0.90:
                        finished = True;

                if cnt_workers<1:
                    if self.logging:
                        if mainnet:
                        	clear_output(wait=True);
                        if mainnet:
                            LOC_MIN, ENERGY_MIN, MALLOB_CHIPS, details = get_status_details_api(JOB_ID);
                        else:
                            LOC_MIN, ENERGY_MIN, MALLOB_CHIPS = 0,0,0;
                            details = "";
                        elapsed_time = time.process_time() - t;
                        # display:
                        table = ([['DYNEXJOB','ELAPSED','WORKERS','CHIPS','✔','STEPS','LOC','✔','ENERGY','✔']]);
                        table.append(['','','*** WAITING FOR READS ***','','','','','','','']);
                        ta = tabulate(table, headers="firstrow", tablefmt='rounded_grid', floatfmt=".2f");
                        print(ta+'\n'+details);
                        
                    time.sleep(2); 

                else:
                    if self.logging:
                        if mainnet:
                        	clear_output(wait=True);
                        if mainnet:
                            LOC_MIN, ENERGY_MIN, MALLOB_CHIPS, details = get_status_details_api(JOB_ID);
                        else:
                            LOC_MIN, ENERGY_MIN, MALLOB_CHIPS = 0,0,0;
                            details = "";
                        elapsed_time = time.process_time() - t;
                        # display:
                        table = ([['DYNEXJOB','ELAPSED','WORKERS','CHIPS','✔','STEPS','LOC','✔','ENERGY','✔']]);
                        table.append([JOB_ID, elapsed_time, cnt_workers, MALLOB_CHIPS, total_chips, total_steps, LOC_MIN, lowest_loc, ENERGY_MIN, lowest_energy]);
                        ta = tabulate(table, headers="firstrow", tablefmt='rounded_grid', floatfmt=".2f");
                        print(ta+'\n'+details);

                        # update mallob - job running: -------------------------------------------------------------------------------------------------
                        if runupdated==False and mainnet:
                            update_job_api(JOB_ID, 1, self.logging);
                            runupdated = True;
                    time.sleep(2);

            # update mallob - job finished: -------------------------------------------------------------------------------------------------
            if mainnet:
                update_job_api(JOB_ID, 2, self.logging, workers=cnt_workers, lowest_loc=lowest_loc, lowest_energy=lowest_energy);

            # update final output (display all workers as stopped as well):
            if cnt_workers>0 and self.logging:
                if mainnet:
                	clear_output(wait=True);
                if mainnet:
                    LOC_MIN, ENERGY_MIN, MALLOB_CHIPS, details = get_status_details_api(JOB_ID, all_stopped = True);
                else:
                    LOC_MIN, ENERGY_MIN, MALLOB_CHIPS = 0,0,0;
                    details = "";
                elapsed_time = time.process_time() - t;
                if mainnet:
                    # display:
                    table = ([['DYNEXJOB','ELAPSED','WORKERS','CHIPS','✔','STEPS','LOC','✔','ENERGY','✔']]);
                    table.append([JOB_ID, elapsed_time, cnt_workers, MALLOB_CHIPS, total_chips, total_steps, LOC_MIN, lowest_loc, ENERGY_MIN, lowest_energy]);
                    ta = tabulate(table, headers="firstrow", tablefmt='rounded_grid', floatfmt=".2f");
                    print(ta+'\n'+details);
                
            elapsed_time = time.process_time() - t
            if self.logging:
                print("[DYNEX] FINISHED READ AFTER","%.2f" % elapsed_time,"SECONDS");

            
            # step 3: now parse voltages: ---------------------------------------------------------------------------------------------------
            
            sampleset = [];
            lowest_energy = 1.7976931348623158e+308;
            lowest_loc = 1.7976931348623158e+308;
            total_chips = 0;
            total_steps = 0;
            lowest_set = [];
            dimod_sample = [];
            for file in files:
                # format: xxx.bin.32.1.0.0.000000
                # jobfile chips steps loc energy
                info = file[len(self.filename)+1:];
                chips = int(info.split(".")[0]);
                steps = int(info.split(".")[1]);
                loc = int(info.split(".")[2]);

                # energy can also be non decimal:
                if len(info.split("."))>4:
                    energy = float(info.split(".")[3]+"."+info.split(".")[4]);
                else:
                    energy = float(info.split(".")[3]);
                total_chips = total_chips + chips;
                total_steps = steps;

                with open(self.filepath+file, 'r') as ffile:
                    data = ffile.read();
                    # enough data?
                    if mainnet:
                    	if len(data)>96:
                    		wallet = data.split("\n")[0];
                    		tmp = data.split("\n")[1];
                    		voltages = tmp.split(", ")[:-1];
                    	else:
                    		voltages = ['NaN']; # invalid file received
                    else: # test-net is not returning wallet
                    	voltages = data.split(", ")[:-1];

                # valid result? ignore Nan values and other incorrect data
                if len(voltages)>0 and voltages[0] != 'NaN' and self.num_variables == len(voltages):
                    sampleset.append(['sample',voltages,'chips',chips,'steps',steps,'falsified softs',loc,'energy',energy]);
                    if loc < lowest_loc:
                        lowest_loc = loc;
                    if energy < lowest_energy:
                        lowest_energy = energy;
                        lowest_set = voltages;
                    # add voltages to dimod return sampleset:
                    dimodsample = {};
                    i = 0;
                    for var in range(0, self.num_variables): 
                        # mapped variable?
                        if var in self.var_mappings:
                            dimodsample[self.var_mappings[var]] = 1; 
                            if (float(voltages[i])<0):
                                dimodsample[self.var_mappings[var]] = 0;  
                        else:
                            dimodsample[i] = 1;  
                            if (float(voltages[i])<0):
                                dimodsample[i] = 0;  
                        i = i + 1
            
                    dimod_sample.append(dimodsample);

                else:
                    print('[DYNEX] OMITTED SOLUTION FILE:',file,' - INCORRECT DATA');

            sampleset.append(['sample',lowest_set,'chips',total_chips,'steps',total_steps,'falsified softs',lowest_loc,'energy',lowest_energy]);
            elapsed_time = time.process_time() - t;

            # delete computing file: ---------------------------------------------------------------------------------------------------
            if mainnet:
            	delete_file_on_ftp(self.ftp_hostname, self.ftp_username, self.ftp_password, self.filepath+self.filename, self.ftp_path, self.logging);
            
            # build sample dict "assignments" with 0/1 and dimod_sampleset ------------------------------------------------------------------
            if self.type == 'wcnf' and len(lowest_set) == self.num_variables:
                sample = {};
                i = 0;
                for var in self.var_mappings:
                    sample[var] = 1;
                    if (float(lowest_set[i])<0):
                        sample[var] = 0;
                    i = i + 1
                self.assignments = sample;

                # generate dimod format sampleset: 
                self.dimod_assignments = dimod.SampleSet.from_samples_bqm(dimod_sample, self.bqm);
                

            if self.logging:
                print("[DYNEX] SAMPLESET LOADED");
            
            # create return sampleset: ------------------------------------------------------------------------------------------------------
            sampleset_clean = [];
            for sample in sampleset:
                sample_dict = Convert(sample);
                sampleset_clean.append(sample_dict);

        except KeyboardInterrupt:
            if mainnet:
                update_job_api(JOB_ID, 2, self.logging);
            print("[DYNEX] Keyboard interrupt");
            return {};

        except Exception as e:
            if mainnet:
                update_job_api(JOB_ID, 2, self.logging);
            print("[DYNEX] Exception encountered:", e);
            return {};

        self.sampleset = sampleset_clean;

        return self.dimod_assignments; 
        
