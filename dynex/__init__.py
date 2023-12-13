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

__version__ = "0.1.11"
__author__ = 'Dynex Developers'
__credits__ = 'Dynex Developers, Contributors, Supporters and the Dynex Community'

# Changelog 0.1.7:
# + all internal funtions: prepend '_x' (and rebuild docs) 
# + don't throw exeception on missing ini, just warning (import dynex issue) 
# + test-net: automatically use max fitting chips; ignore num_reads 
# + remove "boost job priority" for now 

# Changelog 0.1.8:
# + improved accuracy by validating solution file's reported energies with voltages and omitting incorrect reads
# + improved sampling display (showing ground state and decluttered)
# + default logging=False for CQM/BQM/SAT models

# Changelog 0.1.9:
# + moved energy ground state calculation into _DynexSampler class
# + improved debugging=True option for sampling
# + new sampler parameter: bnb=True/False (testnet only; sampling method: branch-and-bound)
# + new function dynex.sample(bqm, **parameters)
# + new function dynex.sample_qubo(Q, offset, **parameters)
# + new function dynex.sample_ising(h, j, **parameters)
# + improved bqm2bin function:
#    - faster: direct conversion from bqm (qubo step omitted)
#    - rydberg hamiltonian formulation
#    - reduction of linear terms

# Changelog 0.1.10:
# + np.float64 conversion bugfix

# Changelog 0.1.11:
# + official Dynex market place version
# + included billing functionality
# + changed dynex.ini for market place compatibility
# + using market place SDK API application layer 
# + file upload / encryption / data handling: server side
# + API layer & AWS elastic cloud support
# + removed file upload progress bar
# + changed refresh interval for display from 2->5 seconds
# + validation clauses
# + progress % and steps during compute
# + new dnx encryption format 

# Upcoming:
# - Multi-model parallel sampling (f.e. for parameter tuning jobs, etc.)


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
import urllib.request, json, urllib.parse
from urllib.error import URLError, HTTPError
import base64
import requests

# Clone sampling:
import multiprocessing
from multiprocessing import Process, Queue

################################################################################################################################
# API FUNCTION CALLS
################################################################################################################################

NUM_RETRIES = 10;

# parse config file:
try:
    config = configparser.ConfigParser();
    config.read('dynex.ini', encoding='UTF-8');
    API_ENDPOINT = config['DYNEX']['API_ENDPOINT']
    API_KEY = config['DYNEX']['API_KEY'];
    API_SECRET = config['DYNEX']['API_SECRET'];
except:
    print('[DYNEX] WARNING: missing configuration file dynex.ini. Please follow the installation instructions at \nhttps://github.com/dynexcoin/DynexSDK/wiki/Installing-the-Dynex-SDK');

def account_status():
    """
    Shows the status of the Dynex SDK account as well as the current (average) block fee for compute:

    .. code-block:: 

        ACCOUNT: <YOUR ACCOUNT IDENTIFICATION>
        API SUCCESSFULLY CONNECTED TO DYNEX
        -----------------------------------
        MAXIMUM NUM_READS: 100,000
        MAXIMUM ANNEALING_TIME: 10,000
        MAXIMUM JOB DURATION: 60 MINUTES
        COMPUTE:
        CURRENT AVG BLOCK FEE: 31.250005004 DNX
        USAGE:
        AVAILABLE BALANCE: 90.0 DNX
        USAGE TOTAL: 0.0 DNX

    """
    
    _check_api_status(logging = True);

def _price_oracle(logging = False):
    """
    `Internal Function`
    
    Dynex API call to output the current average price for compute on Dynex

    :Returns:

    - TRUE if the API call was successful, FALSE if the API call was not successful (`bool`)
    """
    global AVG_BLOCK_FEE
    url = API_ENDPOINT+'/v2/sdk/price_oracle?api_key='+API_KEY+'&api_secret='+API_SECRET;
    with urllib.request.urlopen(url) as ret:
        data = json.load(ret);
    retval = 0;
    if 'error' not in data:
        AVG_BLOCK_FEE = data['avg_block_fee'];
        if logging:
            print('AVERAGE BLOCK FEE:','{:,}'.format(AVG_BLOCK_FEE/1000000000),'DNX');
        retval = AVG_BLOCK_FEE;
    else:
        raise Exception('INVALID API CREDENTIALS');
    return retval;


def _check_api_status(logging = False):
    """
    `Internal Function`
    
    Dynex API call to output the status of the Dynex SDK account

    :Returns:

    - TRUE if the API call was successful, FALSE if the API call was not successful (`bool`)
    """
    
    AVG_BLOCK_FEE = _price_oracle();
    url = API_ENDPOINT+'/v2/sdk/status?api_key='+API_KEY+'&api_secret='+API_SECRET;
    with urllib.request.urlopen(url) as ret:
        data = json.load(ret);
    retval = False;
    if 'error' not in data:
        MAX_CHIPS = data['max_chips'];
        MAX_ANNEALING_TIME = data['max_steps'];
        MAX_DURATION = data['max_duration'];
        TOTAL_USAGE = data['total_usage'];
        CONFIRMED_BALANCE = data['confirmed_balance'];
        ACCOUNT_NAME = data['account_name'];
        if logging:
            print('ACCOUNT:',ACCOUNT_NAME);
            print('API SUCCESSFULLY CONNECTED TO DYNEX');
            print('-----------------------------------');
            print('ACCOUNT LIMITS:');
            print('MAXIMUM NUM_READS:','{:,}'.format(MAX_CHIPS));
            print('MAXIMUM ANNEALING_TIME:','{:,}'.format(MAX_ANNEALING_TIME));
            print('MAXIMUM JOB DURATION:','{:,}'.format(MAX_DURATION),'MINUTES')
            print('COMPUTE:');
            print('CURRENT AVG BLOCK FEE:','{:,}'.format(AVG_BLOCK_FEE/1000000000),'DNX');
            print('USAGE:');
            print('AVAILABLE BALANCE:','{:,}'.format(CONFIRMED_BALANCE/1000000000),'DNX');
            print('USAGE TOTAL:','{:,}'.format(TOTAL_USAGE/1000000000),'DNX');
        retval = True;
    else:
        raise Exception('INVALID API CREDENTIALS');
    return retval;

def _cancel_job_api(JOB_ID, logging = False):
    """
    `Internal Function`
    
    Dynex API call to cancel an ongoing job

    :Returns:

    - TRUE if the API call was successful, FALSE if the API call was not successful (`bool`)
    """
    retval = False;
    url = API_ENDPOINT+'/v2/sdk/job/cancel?api_key='+API_KEY+'&api_secret='+API_SECRET;
    payload = json.dumps({"job_id": JOB_ID});
    
    headers = {
      'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload);
        jsondata = response.json();
        if 'error' in jsondata:
            print('ERROR',jsondata['error']);
            raise Exception(jsondata['error']);
        retval = True;
    
    except HTTPError as e:
        print('[ERROR] Error code: ', e.code)
    except URLError as e:
        print('[ERROR] Reason: ', e.reason)

    return retval;

def _finish_job_api(JOB_ID, MIN_LOC, MIN_ENERGY, logging = False):
    """
    `Internal Function`
    
    Dynex API call to finish an ongoing job

    :Returns:

    - TRUE if the API call was successful, FALSE if the API call was not successful (`bool`)
    """
    retval = False;
    url = API_ENDPOINT+'/v2/sdk/job/finish?api_key='+API_KEY+'&api_secret='+API_SECRET;
    payload = json.dumps({"job_id": JOB_ID, "min_loc": MIN_LOC, "min_energy": MIN_ENERGY});
    
    headers = {
      'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload);
        jsondata = response.json();
        if 'error' in jsondata:
            print('ERROR',jsondata['error']);
            raise Exception(jsondata['error']);
        retval = True;
    
    except HTTPError as e:
        print('[ERROR] Error code: ', e.code)
    except URLError as e:
        print('[ERROR] Reason: ', e.reason)

    return retval;

def _update_job_api(JOB_ID, logging = False):
    """
    `Internal Function`
    
    Dynex API call to cancel an ongoing job

    :Returns:

    - TRUE if the API call was successful, FALSE if the API call was not successful (`bool`)
    """
    retval = False;
    url = API_ENDPOINT+'/v2/sdk/job/update?api_key='+API_KEY+'&api_secret='+API_SECRET;
    payload = json.dumps({"job_id": JOB_ID});
    
    headers = {
      'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload);
        jsondata = response.json();
        if 'error' in jsondata:
            print('ERROR',jsondata['error']);
            raise Exception(jsondata['error']);
        retval = True;
    
    except HTTPError as e:
        print('[ERROR] Error code: ', e.code)
    except URLError as e:
        print('[ERROR] Reason: ', e.reason)

    return retval;

def _post_request(url, opts, file_path):
    opts_json = json.dumps(opts)
    with open(file_path, 'rb') as file:
        files = {
            'opts': (None, opts_json, 'application/json'),
            'job': (file_path, file, 'application/octet-stream')
        }
        response = requests.post(url, files=files)

    return response


def _upload_job_api(sampler, annealing_time, switchfraction, num_reads, alpha=20, beta=20, gamma=1, delta=1, epsilon=1, zeta=1, minimum_stepsize=0.00000006, logging=True, block_fee=0):

    """
    `Internal Function`
    
    Dynex API call to upload a job file and start computing

    :Returns:

    - JOB_ID
    """

    retval = -1;
    filename = '';
    price_per_block = 0;

    # block fee
    if block_fee==0:
        block_fee=_price_oracle();

    print('[DYNEX] AVERAGE BLOCK FEE:','{:,}'.format(block_fee/1000000000),'DNX')

    # parameters:
    url = API_ENDPOINT+'/v2/sdk/job/create?api_key='+API_KEY+'&api_secret='+API_SECRET;

    # options:
    opts = {
            "opts": {
                "annealing_time": annealing_time,
                "switch_fraction": switchfraction,
                "num_reads": num_reads,
                "params":[alpha, beta, gamma, delta, epsilon, zeta],
                "min_step_size": minimum_stepsize,
                "description": sampler.description,
                "block_fee": block_fee
                    }
            };

    # file:
    file_path = sampler.filepath+sampler.filename;

    try:
        response = _post_request(url, opts, file_path);
        jsondata = response.json();
        # error?
        if 'error' in jsondata:
            print("[ERROR]",jsondata['error']);
            raise Exception(jsondata['error']);
        retval = jsondata['job_id'];
        link = jsondata['link'];
        filename = link.split('/')[-1];
        # display applicable block fee:
        price_per_block = jsondata['price_per_block'];
        print('[DYNEX] COST OF COMPUTE:','{:,}'.format(price_per_block/1000000000),'DNX')

    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
        raise SystemExit(errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
        raise SystemExit(errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
        raise SystemExit(errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)
        raise SystemExit(err)

    return retval, filename, price_per_block;


def _get_status_details_api(JOB_ID, annealing_time, all_stopped = False):
    """
    `Internal Function`
    
    Dynex API call to retrieve status of the job

    :Returns:

    - :LOC_MIN: Lowest value of global falsified soft clauses of the problem which is being sampled (`int`)
    
    - :ENERGY_MIN: Lowest QUBO energy of the problem which is being sampled (`double`)
    
    - :CHIPS: The number of chips which are currently sampling (`int`)
    
    - :retval: Tabulated overview of the job status, showing workers, found assignments, etc. (`string`)
    """

    url = API_ENDPOINT+'/v2/sdk/job/atomics?api_key='+API_KEY+'&api_secret='+API_SECRET+'&job_id='+str(JOB_ID);
    
    headers = {
      'Content-Type': 'application/json'
    }
    try:
        response = requests.get(url, headers=headers);
        jsondata = response.json();
        data = jsondata['data'];
        retval = True;
    except HTTPError as e:
        print('[ERROR] Error code: ', e.code)
    except URLError as e:
        print('[ERROR] Reason: ', e.reason)

    table = [['WORKER','VERSION','CHIPS','LOC','ENERGY','RUNTIME','LAST UPDATE','STEPS','STATUS']];
    
    LOC_MIN = 2147483647;
    ENERGY_MIN = 2147483647;
    CHIPS = 0;
    i = 0;
    
    for result in data:
        worker = result['worker_id'][:4]+'..'+result['worker_id'][-4:];
        chips = int(result['chips']);
        loc = int(result['loc']);
        energy = float(result['energy']);
        version = result['version'];
        updated_at = result['updated_at'];
        update_dur = result['update_dur'];
        uptime_dur = result['uptime_dur'];
        steps = int(result['steps']);

        #truncate version
        version = version[:15];

        # update mins:
        if loc < LOC_MIN:
            LOC_MIN = loc;
        if energy < ENERGY_MIN:
            ENERGY_MIN = energy;

        # update number of workers:
        CHIPS = CHIPS + chips;

        # calculate progress:
        progress = 0.0;
        if steps > 0:
            progress = steps / annealing_time * 100;
        steps_str = str(steps) + " ({:.2f}%)".format(progress);

        # status display:
        status = "\033[1;31m%s\033[0m" %'WAITING';
        if int(steps)<int(annealing_time):
            status = "\033[1;32m%s\033[0m" %'RUNNING';
        if int(steps)>=int(annealing_time):
            status = "\033[1;31m%s\033[0m" %'STOPPED';
        if all_stopped:
            status = "\033[1;31m%s\033[0m" %'STOPPED';
            steps = 'STOPPED';

        # add worker information to table:
        if (loc < 2147483647 and energy < 2147483647):
            table.append([worker, version, chips, loc, energy, update_dur, updated_at, steps_str, status]);
        else:
            table.append([worker, version, chips, -1, -1, update_dur, updated_at, steps_str, status]);

        i = i + 1;
    
    # if job not worked on:
    if i==0:
        table.append(['*** WAITING FOR WORKERS ***','','','','','','','','']);
        LOC_MIN = 0;
        ENERGY_MIN = 0;
        CHIPS = 0;

    retval = tabulate(table, headers="firstrow", tablefmt="rounded_grid", stralign="right", floatfmt=".2f")

    return LOC_MIN, ENERGY_MIN, CHIPS, retval;

################################################################################################################################
# TEST dynex.ini CONFIGURATION
################################################################################################################################

def _test_completed():
    """
    `Internal Function`

    :Returns:

    - Returns TRUE if dynex.test() has been successfully completed, FALSE if dynex.test() was not successfully completed (`bool`)
    """
    
    local_path='dynex.test';
    return os.path.isfile(local_path);

def test():
    """
    Performs test of the dynex.ini settings. Successful completion is required to start using the sampler.
    """

    allpassed = True;
    print('[DYNEX] TEST: dimod BQM construction...')
    bqm = dimod.BinaryQuadraticModel({'a': 1.5}, {('a', 'b'): -1}, 0.0, 'BINARY')
    model = BQM(bqm, logging=False);
    print('[DYNEX] PASSED');
    print('[DYNEX] TEST: Dynex Sampler object...')
    sampler = _DynexSampler(model,  mainnet=False, logging=False, test=True);
    print('[DYNEX] PASSED');
    print('[DYNEX] TEST: submitting sample file...')
    worker_user = sampler.solutionuser.split(':')[0]
    worker_pass = sampler.solutionuser.split(':')[1]
    ret = sampler.upload_file_to_ftp(sampler.solutionurl[6:-1], worker_user, worker_pass, sampler.filepath+sampler.filename, '', sampler.logging);
    if ret==False:
        allpassed=False;
        print('[DYNEX] FAILED');
        raise Exception("DYNEX TEST FAILED");
    else:
        print('[DYNEX] PASSED');
    time.sleep(1)
    print('[DYNEX] TEST: retrieving samples...')
    try:
        files = sampler.list_files_with_text();
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

def _check_list_length(lst):
    """
    `Internal Function`

    :Returns:
    - TRUE if the sat problem is k-Sat, FALSE if the problem is 3-sat or 2-sat (`bool`)
    """
    
    for sublist in lst:
        if isinstance(sublist, list) and len(sublist) > 3:
            return True
    return False

# find largest variable in clauses:
def _find_largest_value(lst):
    """
    `Internal Function`

    :Returns:

    - The largest variable in a list of clauses (`int`)
    """
    
    largest_value = None

    for sublist in lst:
        for value in sublist:
            if largest_value is None or value > largest_value:
                largest_value = value

    return largest_value

# create a substitution clause:
def _sat_creator(variables, clause_type, dummy_number, results_clauses):
    """
    `Internal Function`

    Converts a k-sat clause to a number of 3-sat clauses.

    :Parameters:

    - :variables:
    - :clause_type:
    - :dummy_number:
    - :results_clauses:

    :Returns:

    - :dummy_number:
    - :results_clauses:
    
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
def _ksat(clauses):
    """
    `Internal Function`

    Converts a k-sat formulation into 3-sat.

    :Returns:

    - List of clauses of the converted 3-sat (`list`)
    """
    
    results_clauses = [];
    results_clauses.append([1])
    variables = _find_largest_value(clauses);
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
                dummy_number, results_clauses = _sat_creator(first_clause, 1, dummy_number, results_clauses)

                middle_clauses = values[2:len(values)-2]
                dummy_number, results_clauses = _sat_creator(middle_clauses, 2, dummy_number, results_clauses)

                last_clause = values[len(values)-2:]
                dummy_number, results_clauses = _sat_creator(last_clause, 3, dummy_number, results_clauses)

    return results_clauses

################################################################################################################################
# utility functions
################################################################################################################################

def _calculate_sha3_256_hash(string):
    """
    `Internal Function`
    """
    sha3_256_hash = hashlib.sha3_256()
    sha3_256_hash.update(string.encode('utf-8'))
    return sha3_256_hash.hexdigest()

def _calculate_sha3_256_hash_bin(bin):
    """
    `Internal Function`
    """
    sha3_256_hash = hashlib.sha3_256()
    sha3_256_hash.update(bin)
    return sha3_256_hash.hexdigest()

def _Convert(a):
    """
    `Internal Function`
    """
    it = iter(a)
    res_dct = dict(zip(it, it))
    return res_dct;

def _max_value(inputlist):
    """
    `Internal Function`
    """
    return max([sublist[-1] for sublist in inputlist])

def _getCoreCount():
    """
    `Internal Function`
    """
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return (int)(os.popen('grep -c cores /proc/cpuinfo').read())

################################################################################################################################
# save clauses to SAT cnf file
################################################################################################################################

def _save_cnf(clauses, filename, mainnet):
    """
    `Internal Function`

    Saves the model as an encrypted .bin file locally in /tmp as defined in dynex.ini 
    """
    
    num_variables = max(max(abs(lit) for lit in clause) for clause in clauses);
    num_clauses = len(clauses);
    
    with open(filename, 'w') as f:
        line = "p cnf %d %d" % (num_variables, num_clauses);
        
        line_enc = line;
        f.write(line_enc+"\n"); 
        
        for clause in clauses:
            line = ' '.join(str(int(lit)) for lit in clause) + ' 0';
            line_enc = line;
            f.write(line_enc+"\n");

################################################################################################################################
# save wcnf file
################################################################################################################################

def _save_wcnf(clauses, filename, num_variables, num_clauses, mainnet):
    """
    `Internal Function`

    Saves the model as an encrypted .bin file locally in /tmp as defined in dynex.ini 
    """

    with open(filename, 'w') as f:
        line = "p wcnf %d %d" % (num_variables, num_clauses);
        
        line_enc = line;
        f.write(line_enc+"\n"); 

        for clause in clauses:
            line = ' '.join(str(int(lit)) for lit in clause) + ' 0';
        
            line_enc = line;
            f.write(line_enc+"\n"); 

################################################################################################################################
# calculate number of falsified clauses (loc) & Energy based on assignment and model
################################################################################################################################

# moved into class DynexSampler
            
################################################################################################################################
# functions to convert BQM to QUBO
################################################################################################################################

def _max_precision(bqm):
    #return 0.0001;
    max_abs_coeff = np.max(np.abs(bqm.to_numpy_matrix()))
    if max_abs_coeff == 0:
        print('[DYNEX] ERROR: AT LEAST ONE WEIGHT MUST BE > 0.0');
        raise Exception("ERROR: AT LEAST ONE WEIGHT MUST BE > 0.0");
    precision = 10 ** (np.floor(np.log10(max_abs_coeff)) - 4)
    return precision

def _convert_bqm_to_qubo_direct(bqm, relabel=True, logging=True):
    """
    `Internal Function`

    Replacement for _convert_bqm_to_qubo with the following changes:
    - direct conversion from bqm (qubo step omitted), faster
    - rydberg hamiltonian formulation
    - reduction of linear terms
    
    Converts a given Binary Quadratic Model (BQM) problem into a wncf file which is being used by the Dynex platform workers for the sampling process. Every BQM can be converted to a QUBO formulation in polynomial time (and vice-versa) without loss of functionality. During the process, variables are re-labeld and mapped to integer values in the range of [0, NUM_VARIABLES}. The mapping is being made available in sampler.variable_mappings and is used for constructing the returned sampleset object.

    :Notes: 

    - The BQM needs to have at least one defined weight, otherwise an exception is thrown
    - Double values of weights are being converted to integer values with the factor 'PRECISION' 
    - The value for PRECISION is determined automatically with function 10 ** (np.floor(np.log10(max_abs_coeff)) - 4)

    :Parameters:

    - :bqm: the Binary Quadratic Model to be converted (class:`dimod.BinaryQuadraticModel`)

    :Returns:

    - :clauses: A list of all clauses (`list`)
    - :num_variables: number of variables (`int`)
    - :num_clauses: number of clauses (`int`)
    - :mappings: variable mappings original -> integer value (`dict`)
    - :precision: precision of conversion (`double`)
    - :bqm: class:`dimod.BinaryQuadraticModel`

    """
    
    # map variables to integers:
    mappings = bqm.variables._relabel_as_integers();
    
    # define return set:
    clauses = [];

    # linear and quadratic terms of bqm:
    linear = [v for i, v in sorted(bqm.linear.items(), key=lambda x: x[0])];
    quadratic = [[i, j, v] for (i, j), v in bqm.quadratic.items()];
    
    # max precision to be applied:
    precision = _max_precision(bqm);
    
    # max precision is 1:
    if precision>1:
        if logging:
            print("[ÐYNEX] PRECISION CUT FROM",precision,"TO 1");
        precision = 1;

    if logging:
        print("[DYNEX] PRECISION SET TO", precision);

    # linear terms:
    linear_corr = np.round(np.array(linear) / precision);
    _linear = {}
    for i, _ in enumerate(linear):
        if linear_corr[i] == 0:
            continue
        _linear[-np.sign(linear_corr[i]) * (i + 1)] = np.abs(linear_corr[i])
        v = linear_corr[i];
        
    # reduce linear terms:
    _linear_reduced = {}
    for _, i in enumerate(_linear):
        weight = _linear.get(i, 0) - _linear.get(-i, 0)
        if weight > 0:
            _linear_reduced[i] = weight
            clauses.append([weight, i]);
        elif weight < 0:
            _linear_reduced[-i] = -weight
            clauses.append([-weight, -i]);

    num_variables = len(linear);
    
    # quadratic terms:
    if quadratic:
        quadratic_corr = np.round(np.array(quadratic)[:, 2] / precision)
        _quadratic = {}
        for edge, _ in enumerate(quadratic):
            i = quadratic[edge][0] + 1
            j = quadratic[edge][1] + 1

            if quadratic[edge][2] > 0:
                _quadratic[(-i, -j)] = _quadratic.get((i, j), 0) + quadratic_corr[edge]
                v = np.abs(quadratic_corr[edge]);
                if v != 0:
                    clauses.append([v, -i, -j]);
                
            elif quadratic[edge][2] < 0:
                _linear[i] = _linear.get(-i, 0) - quadratic_corr[edge]
                _quadratic[(-i, j)] = _quadratic.get((i, -j), 0) - quadratic_corr[edge]
                v = np.abs(quadratic_corr[edge]);
                if v != 0:
                    clauses.append([v, i, j]);
                    clauses.append([v, -i, j]);
                    clauses.append([v, i, -j]);
                
    # re-map variables:
    bqm.variables._relabel(mappings);
    num_clauses = len(clauses);

    # VALIDATION CLAUSES ===============================
    validation_vars = [1,0,1,0,1,0,1,0];
    validation_weight = 999999;
    for v in range (0, len(validation_vars)):
        dir = 1;
        if validation_vars[v] == 0:
            dir = -1;
        i = num_variables + 1 + v;
        clauses.append([validation_weight, dir * i ]);

    num_variables += len(validation_vars);
    num_clauses = len(clauses);
    # --/ VALIDATION CLAUSES ===========================
    
    return clauses, num_variables, num_clauses, mappings, precision, bqm
    

def _convert_bqm_to_qubo(bqm, relabel=True, logging=True):
    """
    `Internal Function - replaced by _convert_bqm_to_qubo_direct() which uses an enhanced conversion formulation`

    Converts a given Binary Quadratic Model (BQM) problem into a wncf file which is being used by the Dynex platform workers for the sampling process. Every BQM can be converted to a QUBO formulation in polynomial time (and vice-versa) without loss of functionality. During the process, variables are re-labeld and mapped to integer values in the range of [0, NUM_VARIABLES}. The mapping is being made available in sampler.variable_mappings and is used for constructing the returned sampleset object.

    :Notes: 

    - The BQM needs to have at least one defined weight, otherwise an exception is thrown
    - Double values of weights are being converted to integer values with the factor 'PRECISION' 
    - The value for PRECISION is determined automatically with function 10 ** (np.floor(np.log10(max_abs_coeff)) - 4)

    :Parameters:

    - :bqm: the Binary Quadratic Model to be converted (class:`dimod.BinaryQuadraticModel`)

    :Returns:

    - :clauses: A list of all clauses (`list`)
    - :num_variables: number of variables (`int`)
    - :num_clauses: number of clauses (`int`)
    - :mappings: variable mappings original -> integer value (`dict`)
    - :precision: precision of conversion (`double`)
    - :bqm: class:`dimod.BinaryQuadraticModel`

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

    # constant offset:
    W_add = Q[1]; 
    if logging:
        print("[DYNEX] QUBO: Constant offset of the binary quadratic model:", W_add);

    for i in range(0, len(Q_list)):
        touple = Q_list[i];
        i = int(touple[0])+1; # var i; +1 because vars start with 1
        j = int(touple[1])+1; # var j; +1 because vars start with 1
        w = Q[0][touple];     # weight
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
    Creates a model, which can be used by the sampler based on a SAT problem. The Dynex sampler needs a "model" object for sampling. Based on the problem formulation, multiple model classes are supported.

        :Parameters:

        - :clauses: List of sat caluses for this model (`list`)
        - :logging: True to show model creation information, False to silence outputs (`bool`)
        
        :Returns:
        
        - class:`dynex.model`

    :Example:

    Dimod's dimod.binary.BinaryQuadraticModel (BQM) contains linear and quadratic biases for problems formulated as binary quadratic models as well as additional information such as variable labels and offset.

    .. code-block:: Python

        clauses = [[1, -2, 3], [-1, 4, 5], [6, 7, -8], [-9, -10, 11], [12, 13, -14],
           [-1, 15, -16], [17, -18, 19], [-20, 2, 3], [4, -5, 6], [-7, 8, 9],
           [10, 11, -12], [-13, -14, 15], [16, 17, -18], [-19, 20, 1], [2, -3, 4],
           [-5, 6, 7], [8, 9, -10], [-11, -12, 13], [14, 15, -16], [-17, 18, 19]]
        model =  dynex.SAT(clauses)

    """
    def __init__(self, clauses, logging=False):
        self.clauses = clauses;
        self.type = 'cnf';
        self.bqm = "";
        self.logging = logging;
        self.typestr = 'SAT';

class BQM():
    """
    Creates a model, which can be used by the sampler based on a Binary Quadratic Model (BQM) problem. The Dynex sampler needs a "model" object for sampling. Based on the problem formulation, multiple model classes are supported.
    
        :Parameters:

        - :bqm: The BQM to be used for this model (class:`dimod.BinaryQuadraticModel`)
        - :relabel: Defines if the BQM's variable names should be relabeled (`bool`)
        - :logging: True to show model creation information, False to silence outputs (`bool`)
        
        :Returns:
        
        - class:`dynex.model`

    :Example:

    Dimod's `dimod.binary.BinaryQuadraticModel` (BQM) contains linear and quadratic biases for problems formulated as binary quadratic models as well as additional information such as variable labels and offset.

    .. code-block:: Python

        bqm = dimod.BinaryQuadraticModel({'x1': 1.0, 'x2': -1.5, 'x3': 2.0}, 
                                 {('x1', 'x2'): 1.0, ('x2', 'x3'): -2.0}, 
                                 0.0, dimod.BINARY)
        model = dynex.BQM(bqm)

    """
    def __init__(self, bqm, relabel=True, logging=False, formula=2):
        if formula == 1:
            self.clauses, self.num_variables, self.num_clauses, self.var_mappings, self.precision, self.bqm = _convert_bqm_to_qubo(bqm, relabel, logging);
        if formula == 2:
            self.clauses, self.num_variables, self.num_clauses, self.var_mappings, self.precision, self.bqm = _convert_bqm_to_qubo_direct(bqm, relabel, logging);
            
        if self.num_clauses == 0 or self.num_variables == 0:
            raise Exception('[DYNEX] ERROR: Could not initiate model - no variables & clauses');
            return;
        self.type = 'wcnf';
        self.logging = logging;
        self.typestr = 'BQM';

class CQM():
    """
    Creates a model, which can be used by the sampler based on a Constraint Quadratic Model (CQM) problem. The Dynex sampler needs a "model" object for sampling. Based on the problem formulation, multiple model classes are supported.

        :Parameters:

        - :cqm: The BQM to be used for this model (class:`dimod.ConstraintQuadraticModel`)
        - :relabel: Defines if the BQM's variable names should be relabeled (`bool`)
        - :logging: True to show model creation information, False to silence outputs (`bool`)
        
        :Returns:
        
        - class:`dynex.model`

    :Example:

    Dimod's `dimod.ConstrainedQuadraticModel` (CQM) contains linear and quadratic biases for problems formulated as constrained quadratic models as well as additional information such as variable labels, offsets, and equality and inequality constraints.

    .. code-block:: Python

        num_widget_a = dimod.Integer('num_widget_a', upper_bound=7)
        num_widget_b = dimod.Integer('num_widget_b', upper_bound=3)
        cqm = dimod.ConstrainedQuadraticModel()
        cqm.set_objective(-3 * num_widget_a - 4 * num_widget_b)
        cqm.add_constraint(num_widget_a + num_widget_b <= 5, label='total widgets')
        model = dynex.CQM(cqm)


    """
    def __init__(self, cqm, relabel=True, logging=False, formula=2):
        bqm, self.invert = dimod.cqm_to_bqm(cqm)
        if formula == 1:
            self.clauses, self.num_variables, self.num_clauses, self.var_mappings, self.precision, self.bqm = _convert_bqm_to_qubo(bqm, relabel, logging);
        if formula == 2:
            self.clauses, self.num_variables, self.num_clauses, self.var_mappings, self.precision, self.bqm = _convert_bqm_to_qubo_direct(bqm, relabel, logging);
            
        if self.num_clauses == 0 or self.num_variables == 0:
            raise Exception('[DYNEX] ERROR: Could not initiate model - no variables & clauses');
            return;
        self.type = 'wcnf';
        self.logging = logging;
        self.typestr = 'CQM';

################################################################################################################################
# Thread runner: sample clones
################################################################################################################################
def _sample_thread(q, x, model, logging, mainnet, description, num_reads, annealing_time, switchfraction, alpha, beta, gamma, delta, epsilon, zeta, minimum_stepsize, block_fee):
    """
    `Internal Function` which creates a thread for clone sampling
    """
    if logging:
        print('[DYNEX] Clone '+str(x)+' started...'); 
    _sampler = _DynexSampler(model, False, True, description, False);
    _sampleset = _sampler.sample(num_reads, annealing_time, switchfraction, alpha, beta, gamma, delta, epsilon, zeta, minimum_stepsize, False, block_fee);
    if logging:
        print('[DYNEX] Clone '+str(x)+' finished'); 
    q.put(_sampleset);
    return

################################################################################################################################
# Dynex sampling functions
################################################################################################################################
def sample_qubo(Q, offset=0.0, logging=True, formula=2, mainnet=False, description='Dynex SDK Job', test=False, bnb=True, num_reads = 32, annealing_time = 10, clones = 1, switchfraction = 0.0, alpha=20, beta=20, gamma=1, delta=1, epsilon=1, zeta=1, minimum_stepsize = 0.00000006, debugging=False, block_fee=0):
    """
    Samples a Qubo problem.
    
    :Parameters:
    
    - :Q: The Qubo problem
    
    - :offset: The offset value of the Qubo problem
    
    - :logging: Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (`bool`)
    
    - :mainnet: Defines if the mainnet (Dynex platform sampling) or the testnet (local sampling) is being used for sampling (`bool`)
    
    - :description: Defines the description for the sampling, which is shown in Dynex job dashboards as well as in the market place  (`string`)
    
    - :num_reads: Defines the number of parallel samples to be performed (`int` value in the range of [32, MAX_NUM_READS] as defined in your license)

    - :annealing_time: Defines the number of integration steps for the sampling. Models are being converted into neuromorphic circuits, which are then simulated with ODE integration by the participating workers (`int` value in the range of [1, MAX_ANNEALING_TIME] as defined in your license)
        
    - :clones: Defines the number of clones being used for sampling. Default value is 1 which means that no clones are being sampled. Especially when all requested num_reads will fit on one worker, it is desired to also retrieve the optimum ground states found from more than just one worker. The number of clones runs the sampler for n clones in parallel and aggregates the samples. This ensures a broader spectrum of retrieved samples. Please note, it the number of clones is set higher than the number of available threads on your local machine, then the number of clones processed in parallel is being processed in batches. Clone sampling is only available when sampling on the mainnet. (`integer` value in the range of [1,128])

    - :switchfraction: Defines the percentage of variables which are replaced by random values during warm start samplings (`double` in the range of [0.0, 1.0])

    - :alpha: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :beta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :gamma: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :delta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :epsilon: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :zeta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :minimum_stepsize: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is performig adaptive stepsizes for each ODE integration forward Euler step. This value defines the smallest step size for a single adaptive integration step (`double` value in the range of [0.0000000000000001, 1.0])

    - :debugging: Only applicable for test-net sampling. Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (TRUE) (`bool`)

    - :bnb: Use alternative branch-and-bound sampling when in mainnet=False (`bool`)

    - :block_fee: Computing jobs on the Dynex platform are being prioritised by the block fee which is being offered for computation. If this parameter is not specified, the current average block fee on the platform is being charged. To set an individual block fee for the sampling, specify this parameter, which is the amount of DNX in nanoDNX (1 DNX = 1,000,000,000 nanoDNX)

    :Returns:

    - Returns a dimod sampleset object class:`dimod.sampleset`
    
    :Example:

    .. code-block:: Python

        from pyqubo import Array
        N = 15
        K = 3
        numbers = [4.8097315016016315, 4.325157567810298, 2.9877429101815127,
                   3.199880179616316, 0.5787939511978596, 1.2520928214246918,
                   2.262867466401502, 1.2300003067401255, 2.1601079352817925,
                   3.63753899583021, 4.598232793833491, 2.6215815162575646,
                   3.4227134835783364, 0.28254151584552023, 4.2548151473817075]

        q = Array.create('q', N, 'BINARY')
        H = sum(numbers[i] * q[i] for i in range(N)) + 5.0 * (sum(q) - K)**2
        model = H.compile()
        Q, offset = model.to_qubo(index_label=True)
        sampleset = dynex.sample_qubo(Q, offset, formula=2, annealing_time=200, bnb=True)
        print(sampleset)
           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14   energy num_oc.
        0  0  1  0  0  0  0  0  1  0  0  1  0  0  0  0 2.091336       1
        ['BINARY', 1 rows, 1 samples, 15 variables]
    
    """
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset)
    model = BQM(bqm, logging=logging, formula=formula);
    sampler = DynexSampler(model,  mainnet=mainnet, logging=logging, description=description, bnb=bnb);
    sampleset = sampler.sample(num_reads=num_reads, annealing_time=annealing_time, clones=clones, switchfraction=switchfraction, alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon, zeta=zeta, minimum_stepsize=minimum_stepsize, debugging=debugging, block_fee=block_fee);
    return sampleset
    
def sample_ising(h, j, logging=True, formula=2, mainnet=False, description='Dynex SDK Job', test=False, bnb=True, num_reads = 32, annealing_time = 10, clones = 1, switchfraction = 0.0, alpha=20, beta=20, gamma=1, delta=1, epsilon=1, zeta=1, minimum_stepsize = 0.00000006, debugging=False, block_fee=0):
    """
    Samples an Ising problem.
    
    :Parameters:
    
    - :h: Linear biases of the Ising problem
    
    - :j: Quadratic biases of the Ising problem
    
    - :logging: Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (`bool`)
    
    - :mainnet: Defines if the mainnet (Dynex platform sampling) or the testnet (local sampling) is being used for sampling (`bool`)
    
    - :description: Defines the description for the sampling, which is shown in Dynex job dashboards as well as in the market place  (`string`)
    
    - :num_reads: Defines the number of parallel samples to be performed (`int` value in the range of [32, MAX_NUM_READS] as defined in your license)

    - :annealing_time: Defines the number of integration steps for the sampling. Models are being converted into neuromorphic circuits, which are then simulated with ODE integration by the participating workers (`int` value in the range of [1, MAX_ANNEALING_TIME] as defined in your license)
        
    - :clones: Defines the number of clones being used for sampling. Default value is 1 which means that no clones are being sampled. Especially when all requested num_reads will fit on one worker, it is desired to also retrieve the optimum ground states found from more than just one worker. The number of clones runs the sampler for n clones in parallel and aggregates the samples. This ensures a broader spectrum of retrieved samples. Please note, it the number of clones is set higher than the number of available threads on your local machine, then the number of clones processed in parallel is being processed in batches. Clone sampling is only available when sampling on the mainnet. (`integer` value in the range of [1,128])

    - :switchfraction: Defines the percentage of variables which are replaced by random values during warm start samplings (`double` in the range of [0.0, 1.0])

    - :alpha: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :beta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :gamma: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :delta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :epsilon: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :zeta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :minimum_stepsize: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is performig adaptive stepsizes for each ODE integration forward Euler step. This value defines the smallest step size for a single adaptive integration step (`double` value in the range of [0.0000000000000001, 1.0])

    - :debugging: Only applicable for test-net sampling. Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (TRUE) (`bool`)

    - :bnb: Use alternative branch-and-bound sampling when in mainnet=False (`bool`)

    - :block_fee: Computing jobs on the Dynex platform are being prioritised by the block fee which is being offered for computation. If this parameter is not specified, the current average block fee on the platform is being charged. To set an individual block fee for the sampling, specify this parameter, which is the amount of DNX in nanoDNX (1 DNX = 1,000,000,000 nanoDNX)

    :Returns:

    - Returns a dimod sampleset object class:`dimod.sampleset`
    
    """
    bqm = dimod.BinaryQuadraticModel.from_ising(h, j)
    model = BQM(bqm, logging=logging, formula=formula);
    sampler = DynexSampler(model,  mainnet=mainnet, logging=logging, description=description, bnb=bnb);
    sampleset = sampler.sample(num_reads=num_reads, annealing_time=annealing_time, clones=clones, switchfraction=switchfraction, alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon, zeta=zeta, minimum_stepsize=minimum_stepsize, debugging=debugging, block_fee=block_fee);
    return sampleset

def sample(bqm, logging=True, formula=2, mainnet=False, description='Dynex SDK Job', test=False, bnb=True, num_reads = 32, annealing_time = 10, clones = 1, switchfraction = 0.0, alpha=20, beta=20, gamma=1, delta=1, epsilon=1, zeta=1, minimum_stepsize = 0.00000006, debugging=False, block_fee=0):
    """
    Samples a Binary Quadratic Model (bqm).
    
    :Parameters:
    
    - :bqm: Binary quadratic model to sample
    
    - :logging: Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (`bool`)
    
    - :mainnet: Defines if the mainnet (Dynex platform sampling) or the testnet (local sampling) is being used for sampling (`bool`)
    
    - :description: Defines the description for the sampling, which is shown in Dynex job dashboards as well as in the market place  (`string`)
    
    - :num_reads: Defines the number of parallel samples to be performed (`int` value in the range of [32, MAX_NUM_READS] as defined in your license)

    - :annealing_time: Defines the number of integration steps for the sampling. Models are being converted into neuromorphic circuits, which are then simulated with ODE integration by the participating workers (`int` value in the range of [1, MAX_ANNEALING_TIME] as defined in your license)
        
    - :clones: Defines the number of clones being used for sampling. Default value is 1 which means that no clones are being sampled. Especially when all requested num_reads will fit on one worker, it is desired to also retrieve the optimum ground states found from more than just one worker. The number of clones runs the sampler for n clones in parallel and aggregates the samples. This ensures a broader spectrum of retrieved samples. Please note, it the number of clones is set higher than the number of available threads on your local machine, then the number of clones processed in parallel is being processed in batches. Clone sampling is only available when sampling on the mainnet. (`integer` value in the range of [1,128])

    - :switchfraction: Defines the percentage of variables which are replaced by random values during warm start samplings (`double` in the range of [0.0, 1.0])

    - :alpha: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :beta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :gamma: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :delta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :epsilon: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :zeta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :minimum_stepsize: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is performig adaptive stepsizes for each ODE integration forward Euler step. This value defines the smallest step size for a single adaptive integration step (`double` value in the range of [0.0000000000000001, 1.0])

    - :debugging: Only applicable for test-net sampling. Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (TRUE) (`bool`)

    - :bnb: Use alternative branch-and-bound sampling when in mainnet=False (`bool`)

    - :block_fee: Computing jobs on the Dynex platform are being prioritised by the block fee which is being offered for computation. If this parameter is not specified, the current average block fee on the platform is being charged. To set an individual block fee for the sampling, specify this parameter, which is the amount of DNX in nanoDNX (1 DNX = 1,000,000,000 nanoDNX)

    :Returns:

    - Returns a dimod sampleset object class:`dimod.sampleset`
    
    :Example:

    .. code-block:: Python

        from pyqubo import Array
        N = 15
        K = 3
        numbers = [4.8097315016016315, 4.325157567810298, 2.9877429101815127,
                   3.199880179616316, 0.5787939511978596, 1.2520928214246918,
                   2.262867466401502, 1.2300003067401255, 2.1601079352817925,
                   3.63753899583021, 4.598232793833491, 2.6215815162575646,
                   3.4227134835783364, 0.28254151584552023, 4.2548151473817075]

        q = Array.create('q', N, 'BINARY')
        H = sum(numbers[i] * q[i] for i in range(N)) + 5.0 * (sum(q) - K)**2
        model = H.compile()
        Q, offset = model.to_qubo(index_label=True)
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset)
        sampleset = dynex.sample(bqm, offset, formula=2, annealing_time=200, bnb=True)
        print(sampleset)
           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14   energy num_oc.
        0  0  1  0  0  0  0  0  1  0  0  1  0  0  0  0 2.091336       1
        ['BINARY', 1 rows, 1 samples, 15 variables]
    
    """
    model = BQM(bqm, logging=logging, formula=formula);
    sampler = DynexSampler(model,  mainnet=mainnet, logging=logging, description=description, bnb=bnb);
    sampleset = sampler.sample(num_reads=num_reads, annealing_time=annealing_time, clones=clones, switchfraction=switchfraction, alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon, zeta=zeta, minimum_stepsize=minimum_stepsize, debugging=debugging, block_fee=block_fee);
    return sampleset

        
################################################################################################################################
# Dynex Sampler (public class)
################################################################################################################################
class DynexSampler:
    """
    Initialises the sampler object given a model.

    :Parameters:

    - :logging: Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (`bool`)
    - :mainnet: Defines if the mainnet (Dynex platform sampling) or the testnet (local sampling) is being used for sampling (`bool`)
    - :description: Defines the description for the sampling, which is shown in Dynex job dashboards as well as in the market place  (`string`)

    :Returns:

    - class:`dynex.samper`

    :Example:

    .. code-block:: Python

        sampler = dynex.DynexSampler(model)

    """
    def __init__(self, model, logging=True, mainnet=True, description='Dynex SDK Job', test=False, bnb=True):
        
        # multi-model parallel sampling
        if isinstance(model, list):
            if mainnet==False:
                raise Exception("[ÐYNEX] ERROR: Multi model parallel sampling is only supported on mainnet");
            if logging:
                print("[ÐYNEX] MULTI-MODEL PARALLEL SAMPLING:",len(model),'MODELS');

        self.state = 'initialised';
        self.model = model;
        self.logging = logging;
        self.mainnet = mainnet;
        self.description = description;
        self.test = test;
        self.dimod_assignments = {};
        self.bnb = bnb; 
        
    def sample(self, num_reads = 32, annealing_time = 10, clones = 1, switchfraction = 0.0, alpha=20, beta=20, gamma=1, delta=1, epsilon=1, zeta=1, minimum_stepsize = 0.00000006, debugging=False, block_fee=0):
        """
        The main sampling function:

        :Parameters:

        - :num_reads: Defines the number of parallel samples to be performed (`int` value in the range of [32, MAX_NUM_READS] as defined in your license)

        - :annealing_time: Defines the number of integration steps for the sampling. Models are being converted into neuromorphic circuits, which are then simulated with ODE integration by the participating workers (`int` value in the range of [1, MAX_ANNEALING_TIME] as defined in your license)
        
        - :clones: Defines the number of clones being used for sampling. Default value is 1 which means that no clones are being sampled. Especially when all requested num_reads will fit on one worker, it is desired to also retrieve the optimum ground states found from more than just one worker. The number of clones runs the sampler for n clones in parallel and aggregates the samples. This ensures a broader spectrum of retrieved samples. Please note, it the number of clones is set higher than the number of available threads on your local machine, then the number of clones processed in parallel is being processed in batches. Clone sampling is only available when sampling on the mainnet. (`integer` value in the range of [1,128])

        - :switchfraction: Defines the percentage of variables which are replaced by random values during warm start samplings (`double` in the range of [0.0, 1.0])

        - :alpha: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

        - :beta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

        - :gamma: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

        - :delta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

        - :epsilon: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

        - :zeta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

        - :minimum_stepsize: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is performig adaptive stepsizes for each ODE integration forward Euler step. This value defines the smallest step size for a single adaptive integration step (`double` value in the range of [0.0000000000000001, 1.0])

        - :debugging: Only applicable for test-net sampling. Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (TRUE) (`bool`)

        - :bnb: Use alternative branch-and-bound sampling when in mainnet=False (`bool`)

        - :block_fee: Computing jobs on the Dynex platform are being prioritised by the block fee which is being offered for computation. If this parameter is not specified, the current average block fee on the platform is being charged. To set an individual block fee for the sampling, specify this parameter, which is the amount of DNX in nanoDNX (1 DNX = 1,000,000,000 nanoDNX) 

        :Returns:

        - Returns a dimod sampleset object class:`dimod.sampleset`

        :Example:

        .. code-block:: 

            import dynex
            import dimod

            # Define the QUBU problem:
            bqmodel = dimod.BinaryQuadraticModel({0: -1, 1: -1}, {(0, 1): 2}, 0.0, dimod.BINARY)  

            # Sample the problem:
            model = dynex.BQM(bqmodel)
            sampler = dynex.DynexSampler(model)
            sampleset = sampler.sample(num_reads=32, annealing_time = 100)

            # Output the result:
            print(sampleset)

        .. code-block:: 

            ╭────────────┬───────────┬───────────┬─────────┬─────┬─────────┬───────┬─────┬──────────┬──────────╮
            │   DYNEXJOB │   ELAPSED │   WORKERS │   CHIPS │   ✔ │   STEPS │   LOC │   ✔ │   ENERGY │        ✔ │
            ├────────────┼───────────┼───────────┼─────────┼─────┼─────────┼───────┼─────┼──────────┼──────────┤
            │       3617 │      0.07 │         1 │       0 │  32 │     100 │     0 │   1 │        0 │ 10000.00 │
            ╰────────────┴───────────┴───────────┴─────────┴─────┴─────────┴───────┴─────┴──────────┴──────────╯
            ╭─────────────────────────────┬───────────┬─────────┬───────┬──────────┬───────────┬───────────────┬──────────╮
            │                      WORKER │   VERSION │   CHIPS │   LOC │   ENERGY │   RUNTIME │   LAST UPDATE │   STATUS │
            ├─────────────────────────────┼───────────┼─────────┼───────┼──────────┼───────────┼───────────────┼──────────┤
            │ *** WAITING FOR WORKERS *** │           │         │       │          │           │               │          │
            ╰─────────────────────────────┴───────────┴─────────┴───────┴──────────┴───────────┴───────────────┴──────────╯
            [DYNEX] FINISHED READ AFTER 0.07 SECONDS
            [DYNEX] PARSING 1 VOLTAGE ASSIGNMENT FILES...
            progress: 100%
            1/1 [00:05<00:00, 5.14s/it]
            [DYNEX] SAMPLESET LOADED
            [DYNEX] MALLOB: JOB UPDATED: 3617 STATUS: 2
               0  1 energy num_oc.
            0  0  1   -1.0       1
            ['BINARY', 1 rows, 1 samples, 2 variables]
        """

        # assert parameters:
        if clones < 1:
            raise Exception("[DYNEX] ERROR: Value of clones must be in range [1,128]");
        if clones > 128:
            raise Exception("[DYNEX] ERROR: Value of clones must be in range [1,128]");
        if self.mainnet==False and clones > 1:
            raise Exception("[DYNEX] ERROR: Clone sampling is only supported on the mainnet");
        
        # sampling without clones: -------------------------------------------------------------------------------------------
        if clones == 1:
            _sampler = _DynexSampler(self.model, self.logging, self.mainnet, self.description, self.test, self.bnb);
            _sampleset = _sampler.sample(num_reads, annealing_time, switchfraction, alpha, beta, gamma, delta, epsilon, zeta, minimum_stepsize, debugging, block_fee);
            return _sampleset;
        
        # sampling with clones: ----------------------------------------------------------------------------------------------
        else:
            supported_threads = _getCoreCount() * 2;
            if clones > supported_threads:
                print('[DYNEX] WARNING: number of clones > CPU cores: clones:',clones,' threads available:',supported_threads);
                
            jobs = [];
            results = [];

            if self.logging:
                print('[DYNEX] STARTING SAMPLING (',clones,'CLONES )...');
    
            # define n samplers:
            for i in range(clones):
                q = Queue()
                results.append(q)
                p = multiprocessing.Process(target=_sample_thread, args=(q, i, self.model, self.logging, self.mainnet, self.description, num_reads, annealing_time, switchfraction, alpha, beta, gamma, delta, epsilon, zeta, minimum_stepsize, block_fee))
                jobs.append(p)
                p.start()

            # wait for samplers to finish:
            for job in jobs:
                job.join()
    
            # collect samples for each job:
            assignments_cum = [];
            for result in results:
                assignments = result.get();
                assignments_cum.append(assignments);
    
            # accumulate and aggregate all results:
            r = None;
            for assignment in assignments_cum:
                if len(assignment)>0:
                    if r == None:
                        r = assignment;
                    else:
                        r = dimod.concatenate((r,assignment))

            # aggregate samples:
            r = r.aggregate() 
            
            self.dimod_assignments = r;
            
            return r

    

################################################################################################################################
# Dynex Sampler class (private)
################################################################################################################################

class _DynexSampler:
    """
    `Internal Class` which is called by public class `DynexSampler`
    """
    def __init__(self, model, logging=True, mainnet=True, description='Dynex SDK Job', test=False, 
                    bnb=True): 
        
        if not test and not _test_completed():
            raise Exception("CONFIGURATION TEST NOT COMPLETED. PLEASE RUN 'dynex.test()'");

        self.description = description;
        
        # parse config file:
        config = configparser.ConfigParser();
        config.read('dynex.ini', encoding='UTF-8');
        
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
        self.bnb = bnb;

        # multi-model parallel sampling?
        multi_model_mode = False;
        if isinstance(model, list):
            if mainnet == False:
                raise Exception("[ÐYNEX] ERROR: Multi model parallel sampling is only supported on mainnet");
            multi_model_mode = True;

        self.multi_model_mode = multi_model_mode;
            
        #single model sampling:
        if multi_model_mode == False:
            # auto generated temp filename:
            self.filename = secrets.token_hex(16)+".dnx";
            self.logging = logging;
            self.mainnet = mainnet;
            self.typestr = model.typestr;
            
            if model.type == 'cnf':
                # convert to 3sat?
                if (_check_list_length(model.clauses)):
                    # we need to convert to 3sat:
                    self.clauses = _ksat(model.clauses);
                else:
                    self.clauses = model.clauses;
                _save_cnf(self.clauses, self.filepath+self.filename, mainnet);
                self.num_clauses = len(self.clauses);
                self.num_variables = _max_value(self.clauses) - 1;
            
            if model.type == 'wcnf':
                self.clauses = model.clauses;
                self.num_variables = model.num_variables;
                self.num_clauses = model.num_clauses;
                self.var_mappings = model.var_mappings;
                self.precision = model.precision;
                _save_wcnf(self.clauses, self.filepath+self.filename, self.num_variables, self.num_clauses, mainnet); 

            self.type = model.type;
            self.assignments = {};
            self.dimod_assignments = {};
            self.bqm = model.bqm;
            self.model = model;

        # multi model sampling:
        else:
            _filename = [];
            _typestr = [];
            _clauses = [];
            _num_clauses = [];
            _num_variables = [];
            _var_mappings = [];
            _precision = [];
            _type = [];
            _assignments = [];
            _dimod_assignments = [];
            _bqm = [];
            _model = [];
            for m in model:
                _filename.append(secrets.token_hex(16)+".dnx");
                _typestr.append(m.type);
                if m.type == 'cnf':
                    raise Exception("[ÐYNEX] ERROR: Multi model parallel sampling is currently not implemented for SAT");
                if m.type == 'wcnf':
                    _clauses.append(m.clauses);
                    _num_clauses.append(m.num_clauses);
                    _num_variables.append(m.num_variables);
                    _var_mappings.append(m.var_mappings);
                    _precision.append(m.precision);
                    _save_wcnf(_clauses[-1], self.filepath+_filename[-1], _num_variables[-1], _num_clauses[-1], mainnet); 
                _type.append(m.type);
                _assignments.append({});
                _dimod_assignments.append({});
                _bqm.append(m.bqm);
                _model.append(m);
            self.filename = _filename;
            self.typestr = _typestr;
            self.clauses = _clauses;
            self.num_clauses = _num_clauses;
            self.num_variables = _num_variables;
            self.var_mappings = _var_mappings;
            self.precision = _precision;
            self.type = _type;
            self.assignments = _assignments;
            self.dimod_assignments = _dimod_assignments;
            self.bqm = _bqm;
            self.model = _model;
            self.logging = logging;
            self.mainnet = mainnet;


        if self.logging:
            print("[DYNEX] SAMPLER INITIALISED")
            
    # deletes all assignment files on FTP
    def cleanup_ftp(self, files):
        """
        `Internal Function`

        This function is called on __exit__ of the sampler class or by sampler.clear(). 
        It ensures that submitted sample-files, which have not been parsed and used from the sampler, will be deleted on the FTP server. 
        """

        if len(files)>0:
            try:
                host = self.solutionurl[6:-1];
                username = self.solutionuser.split(":")[0];
                password = self.solutionuser.split(":")[1]; 
                directory = "";
                ftp = FTP(host);
                ftp.login(username, password);
                ftp.cwd(directory);
                for file in files:
                    ftp.delete(file);
                if self.logging:
                    print("[ÐYNEX] FTP DATA CLEANED");
            except Exception as e:
                print(f"[DYNEX] An error occurred while deleting file: {str(e)}")
                raise Exception("ERROR: An error occurred while deleting file");
            finally:
                ftp.quit();
        return;
     
    # delete file from FTP server
    def delete_file_on_ftp(self, hostname, username, password, local_file_path, remote_directory, logging=True):
        """
        `Internal Function`

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
            
    # upload file to ftp server
    def upload_file_to_ftp(self, hostname, username, password, local_file_path, remote_directory, logging=True):
        """
        `Internal Function`

        Submits a computation file (xxx.bin) to the FTP server as defined in dynex.ini

        :Returns:

        - Status if successul or failed (`bool`)
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

                # upload:
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
            raise Exception("ERROR: An error occurred while sending the file");
            retval = False;
        finally:
            ftp.quit();
        return retval;

    # calculate ground state energy and numer of falsified softs from model ==========================================================
    def _energy(self, sample, mapping=True):
        """
        `Internal Function`
        
        Takes a model and dimod samples and calculates the energy and loc.
        
        Input: 
        ======
        
        - dimod sample (dict) with mapping = True
          example: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0}
          
        or
          
        - assignments (list) with mapping = False (raw solution file)
          example: [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1]
        """
        # convert dimod sample to wcnf mapping:
        wcnf_vars = []; 
        if mapping:
            for v in sample:
                if v in self.model.var_mappings:
                    v_mapped = self.model.var_mappings[v];
                else:
                    v_mapped = v;
                wcnf_vars.append(sample[v_mapped])
        # or convert solution file to 0/1:
        else:
            for v in sample:
                if v>0:
                    wcnf_vars.append(1);
                else:
                    wcnf_vars.append(0);
            
        loc = 0;
        energy = 0.0;
        for clause in self.model.clauses:
            
            if len(clause)==2:
                # 2-lit clause:
                w = clause[0];
                i = int(abs(clause[1]));
                i_dir = np.sign(clause[1]);
                if i_dir == -1:
                    i_dir = 0;
                i_assign = wcnf_vars[i-1];
                if (i_dir != i_assign):
                    loc += 1;
                    energy += w;
            else:
                # 3-lit clause:
                w = clause[0];
                i = int(abs(clause[1]));
                i_dir = np.sign(clause[1]);
                if i_dir == -1:
                    i_dir = 0;
                i_assign = wcnf_vars[i-1];
                j = int(abs(clause[2]));
                j_dir = np.sign(clause[2]);
                if j_dir == -1:
                    j_dir = 0;
                j_assign = wcnf_vars[j-1];
                if (i_dir != i_assign) and (j_dir != j_assign):
                    loc += 1;
                    energy += w;
                
        return loc, energy
            
    # list local available (downloaded) iles in /tmp =================================================================================
    def list_files_with_text_local(self):
        """
        `Internal Function`

        Scans the temporary directory for assignment files

        :Returns:

        - Returns a list of all assignment files (filenames) which are locally available in /tmp as specified in dynex.ini for the current sampler model (`list`)
        """

        directory = self.filepath_full; 
        fn = self.filename+".";
        # list to store files
        filtered_files = []

        # search for current solution files:
        for filename in os.listdir(directory):
            if filename.startswith(fn):
                if os.path.getsize(directory+'/'+filename)>0:
                    filtered_files.append(filename)

        return filtered_files; 
    
    # verify correctness of downloaded file (loc and energy) ==========================================================================
    def validate_file(self, file, debugging=False):
        """
        `Internal Function`
        
        Validates loc and energy provided in filename with voltages. File not matching will be deleted on FTP and locally.
        """
        valid = False;
        
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
        
        with open(self.filepath+file, 'r') as ffile:
            data = ffile.read();
            # enough data?
            if self.mainnet:
                if len(data)>96:
                    wallet = data.split("\n")[0];
                    tmp = data.split("\n")[1];
                    voltages = tmp.split(", ")[:-1];
                else:
                    voltages = ['NaN']; # invalid file received
            else: # test-net is not returning wallet
                voltages = data.split(", ")[:-1];
                
            # convert string voltages to list of floats:
            voltages = list(map(float, voltages));
            if debugging:
                print('DEBUG:');
                print(voltages);

            # valid result? ignore Nan values and other incorrect data
            if len(voltages)>0 and voltages[0] != 'NaN' and self.num_variables == len(voltages):
                val_loc, val_energy = self._energy(voltages, mapping=False);
                
                # from later versions onwards, enforce also correctness of LOC (TBD):
                if energy == val_energy:
                    valid = True;

                if debugging:
                    print('DEBUG:',self.filename,chips,steps,loc,energy,'=>',val_loc, val_energy, 'valid?',valid);

            else:
                if debugging:
                    print('DEBUG:',self.filename,' NaN or num_variables =',len(voltages),' vs ',self.num_variables, 'valid?',valid);
                        
        return valid;
        

    # list and download solution files ================================================================================================
    def list_files_with_text(self, debugging=False):
        """
        `Internal Function`

        Downloads assignment files from the FTP server specified in dynex.ini and stores them in /tmp as specified in dynex.ini
        Downloaded files are automatically deleted on the FTP server.

        :Returns:

        - List of locally in /tmp saved assignment files for the current sampler model (`list`)
        """
        
        host = self.solutionurl[6:-1];
        username = self.solutionuser.split(":")[0];
        password = self.solutionuser.split(":")[1]; 
        directory = "";
        text = self.filename;
        # Connect to the FTP server
        ftp = FTP(host)
        ftp.login(username, password)

        # Change to the specified directory
        ftp.cwd(directory)

        # List all (fully uploaded) files in the directory (minimum size)
        target_size = 97 + self.num_variables;
        for name, facts in ftp.mlsd(): 
            if 'size' in facts:
                if int(facts['size'])>=target_size and name.startswith(text):
                    # download file if not already local:
                    local_path = self.filepath+name;
                    if os.path.isfile(local_path)==False or os.path.getsize(local_path)==0:
                        with open(local_path, 'wb') as file:
                            ftp.retrbinary('RETR ' + name, file.write); 
                            file.close();
                        # correct file?
                        if self.validate_file(name, debugging)==False:
                            if self.logging:
                                print('[DYNEX] REMOVING SOLUTION FILE',name,'(WRONG ENERGY REPORTED OR INCORRECT VOLTAGES)');
                            os.remove(local_path);
                            ftp.delete(name);
                        else:
                            # correctly downloaded?
                            cnt = 0;
                            while os.path.getsize(local_path)==0:
                                time.sleep(1);
                                with open(local_path, 'wb') as file:
                                    if self.logging:
                                        print('[DYNEX] REDOWNLOADING FILE',name);
                                    ftp.retrbinary('RETR ' + name, file.write);
                                    file.close();
                                # correct file?
                                if self.validate_file(name, debugging) == False:
                                    if self.logging:
                                        print('[DYNEX] REMOVING SOLUTION FILE',name,'(WRONG ENERGY REPORTED OR INCORRECT VOLTAGES)');
                                    os.remove(local_path);
                                    break;
                                cnt += 1;
                                if cnt>=10:
                                    break;
                            # finally we delete downloaded files from FTP:
                            ftp.delete(name); 

        # Close the FTP connection
        ftp.quit();

        # In our status view, we show the local, downloaded and available files:
        filtered_files = self.list_files_with_text_local();

        return filtered_files
    
    # clean function ======================================================================================================================
    def _clean(self):
        """
        `Internal Function` 
        This function can be called after finishing a sampling process on the Mainnet. It ensures that submitted sample-files,
        which have not been parsed and used from the sampler, will be deleted on the FTP server. It is also called automatically 
        during __exit___ event of the sampler class.
        """
        if self.mainnet:
            files = self.list_files_with_text(); 
            self.cleanup_ftp(files);

    # on exit ==============================================================================================================================
    def __exit__(self, exc_type, exc_value, traceback):
        """
        `Internal Function` 
        Upon __exit__, the function clean() is being called.
        """
        print('[DYNEX] SAMPLER EXIT');
        
    # update function: =====================================================================================================================
    def _update(self, model, logging=True):
        """
        `Internal Function` 
        Typically, the sampler object is being initialised with a defined model class. This model can also be updated without
        regenerating a new sampler object by calling the function update(model).
        """
        self.logging = logging;
        self.filename     = secrets.token_hex(16)+".bin"; 
        
        if model.type == 'cnf':
            # convert to 3sat?
            if (_check_list_length(model.clauses)):
                self.clauses = _ksat(model.clauses);
            else:
                self.clauses = model.clauses;
            _save_cnf(self.clauses, self.filepath+self.filename);
        
        if model.type == 'wcnf':
            self.clauses = model.clauses;
            self.num_variables = model.num_variables;
            self.num_clauses = model.num_clauses;
            self.var_mappings = model.var_mappings;
            self.precision = model.precision;
            _save_wcnf(self.clauses, self.filepath+self.filename, self.num_variables, self.num_clauses, self.mainnet); 
        
        self.type = model.type;
        self.assignments = {};
        self.dimod_assignments = {};
        self.bqm = model.bqm;

    # print summary of sampler: =============================================================================================================
    def _print(self):
        """
        `Internal Function` 
        Prints summary information about the sampler object:

        - :Mainnet: If the mainnet (Dynex platform sampling) or the testnet (local sampling) is being used for sampling (`bool`)
        - :logging: Show progress and status information or be quiet and omit terminal output (`bool`)
        - :tmp filename: The filename of the computation file (`string`)
        - :model type: [cnf, wcnf]: The type of the model: Sat problems (cnf) or QUBU/Ising type problems (wcnf) (`string`)
        - :num_variables: The number of variables of the model (`int`)
        - :num_clauses: The number of clauses of the model (`int`)

        :Example:

        .. code-block:: 

            DynexSampler object
            mainnet? True
            logging? True
            tmp filename: tmp/b8fa34a815f96098438d68142dfb68b6.dnx
            model type: BQM
            num variables: 15
            num clauses: 120
            configuration: dynex.ini
        """
        print('{DynexSampler object}');
        print('mainnet?', self.mainnet);
        print('logging?', self.logging);
        print('tmp filename:',self.filepath+self.filename);
        print('model type:', self.typestr);
        print('num variables:', self.num_variables);
        print('num clauses:', self.num_clauses);
        print('configuration: dynex.ini');

    # convert a sampler.sampleset[x]['sample'] into an assignment: ==========================================================================
    def _sample_to_assignments(self, lowest_set):
        """
        `Internal Function` 
        The voltates of a sampling can be retrieved from the sampler with sampler.sampleset

        The sampler.sampleset returns a list of voltages for each variable, ranging from -1.0 to +1.0 and is a double precision value. Sometimes it is required to transform these voltages to binary values 0 (for negative voltages) or 1 (for positive voltages). This function converts a given sampler.sampleset[x] from voltages to binary values.

        :Parameters:

        - :lowest_set: The class:`dynex.sampler.assignment' which has to be converted (`list`)

        :Returns:

        - Returns the converted sample as `list`
        """
        sample = {};
        i = 0;
        for var in self.var_mappings:
            sample[var] = 1;
            if (float(lowest_set[i])<0):
                sample[var] = 0;
            i = i + 1
        return sample;
    
    # sampling entry point: =================================================================================================================
    def sample(self, num_reads = 32, annealing_time = 10, switchfraction = 0.0, alpha=20, beta=20, gamma=1, delta=1, epsilon=1, zeta=1, minimum_stepsize = 0.00000006, debugging=False, block_fee=0):
        """
        `Internal Function` which is called by public function `DynexSampler.sample` 
        """
        
        retval = {};

        # In a malleable environment, it is rarely possible that a worker is submitting an inconsistent solution file. If the job
        # is small, we need to re-sample again. This routine samples up to NUM_RETRIES (10) times. If an error occurs, or
        # a keyboard interrupt was triggered, the return value is a dict containing key 'error'
        
        for i in range(0, NUM_RETRIES):
            retval = self._sample(num_reads, annealing_time, switchfraction, alpha, beta, gamma, delta, epsilon, zeta, minimum_stepsize, debugging, block_fee);
            if len(retval)>0:
                break;
            
            # TODO: support multi-model sampling
            print("[DYNEX] NO VALID SAMPLE RESULT FOUND. RESAMPLING...", i+1,'/',NUM_RETRIES)
            time.sleep(2);
            # generate a fresh sampling file:
            self.filename = secrets.token_hex(16)+".bin";
            if self.model.type == 'cnf':
                # convert to 3sat?
                if (_check_list_length(self.model.clauses)):
                    self.clauses = _ksat(self.model.clauses);
                else:
                    self.clauses = self.model.clauses;
                _save_cnf(self.clauses, self.filepath+self.filename);
            if self.model.type == 'wcnf':
                _save_wcnf(self.clauses, self.filepath+self.filename, self.num_variables, self.num_clauses, self.mainnet); 

        # aggregate sampleset:
        if len(retval)>0 and ('error' in retval) == False:
            retval = retval.aggregate();
            
        return retval
    
    # main sampling function =================================================================================================================
    def _sample(self, num_reads = 32, annealing_time = 10, switchfraction = 0.0, alpha=20, beta=20, gamma=1, delta=1, epsilon=1, zeta=1, minimum_stepsize = 0.00000006, debugging=False, block_fee=0):
        """
        `Internal Function` which is called by private function `DynexSampler.sample`. This functions performs the sampling. 
        """
        
        if self.multi_model_mode == True:
            raise Exception('ERROR: Multi-model parallel sampling is not implemented yet');

        mainnet = self.mainnet;
        price_per_block = 0;

        try:
        
            # step 1: upload problem file to Dynex Platform: ---------------------------------------------------------------------------------
            if mainnet:
                # create job on mallob system:
                JOB_ID, self.filename, price_per_block = _upload_job_api(self, annealing_time, switchfraction, num_reads, alpha, beta, gamma, delta, epsilon, zeta, minimum_stepsize, self.logging, block_fee);

                # show effective price in DNX:
                price_per_block = price_per_block/1000000000;
                
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
                # in test-net, it cannot guaranteed that all requested chips are fitting:
                num_reads = 0;
                
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

                # use branch-and-bound (testnet) sampler instead?:
                if self.bnb:
                    command = self.solverpath+"dynex-testnet-bnb "+self.filepath_full+"/"+self.filename;
                
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

            # initialise display:
            if mainnet and debugging==False:
                clear_output(wait=True);
                table = ([['DYNEXJOB', 'BLOCK FEE', 'ELAPSED','WORKERS READ','CHIPS','STEPS','GROUND STATE']]);
                table.append(['','','','*** WAITING FOR READS ***','','','']);
                ta = tabulate(table, headers="firstrow", tablefmt='rounded_grid', floatfmt=".2f");
                print(ta+'\n');

            while finished==False:
                total_chips = 0;
                total_steps = 0;
                lowest_energy = 1.7976931348623158e+308;
                lowest_loc = 1.7976931348623158e+308;

                # retrieve solutions
                if mainnet:
                    try:
                        files = self.list_files_with_text(debugging);
                        cnt_workers = len(files);
                    except Exception as e:
                        print('[DYNEX] CONNECTION TO FTP ENDPOINT FAILED:',e);
                        raise Exception('ERROR: CONNECTION TO FTP ENDPOINT FAILED')
                        files = []; 
                else:
                    files = self.list_files_with_text_local(); 
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
                        if mainnet and debugging==False:
                            clear_output(wait=True);
                        if mainnet: 
                            LOC_MIN, ENERGY_MIN, MALLOB_CHIPS, details = _get_status_details_api(JOB_ID, annealing_time);
                        else:
                            LOC_MIN, ENERGY_MIN, MALLOB_CHIPS = 0,0,0;
                            details = "";
                        elapsed_time = time.process_time() - t;
                        # display:
                        table = ([['DYNEXJOB','BLOCK FEE','ELAPSED','WORKERS READ','CHIPS','STEPS','GROUND STATE']]);
                        table.append([JOB_ID, price_per_block,'','*** WAITING FOR READS ***','','','']);
                        ta = tabulate(table, headers="firstrow", tablefmt='rounded_grid', floatfmt=".2f");
                        print(ta+'\n'+details);
                        
                    time.sleep(2); 

                else:
                    if self.logging:
                        if mainnet and debugging==False:
                            clear_output(wait=True);
                        if mainnet:
                            LOC_MIN, ENERGY_MIN, MALLOB_CHIPS, details = _get_status_details_api(JOB_ID, annealing_time);
                        else:
                            LOC_MIN, ENERGY_MIN, MALLOB_CHIPS = 0,0,0;
                            details = "";
                        elapsed_time = time.process_time() - t;
                        # display:
                        table = ([['DYNEXJOB','BLOCK FEE','ELAPSED','WORKERS READ','CHIPS','STEPS','GROUND STATE']]);
                        table.append([JOB_ID, price_per_block, elapsed_time, cnt_workers, total_chips, total_steps, lowest_energy]);
                        
                        ta = tabulate(table, headers="firstrow", tablefmt='rounded_grid', floatfmt=".2f");
                        print(ta+'\n'+details);

                        # update mallob - job running: -------------------------------------------------------------------------------------------------
                        if runupdated==False and mainnet:
                            _update_job_api(JOB_ID, self.logging);
                            runupdated = True;
                    time.sleep(5);

            # update mallob - job finished: -------------------------------------------------------------------------------------------------
            if mainnet:
                _finish_job_api(JOB_ID, lowest_loc, lowest_energy, self.logging);
                #_update_job_api(JOB_ID, 2, self.logging, workers=cnt_workers, lowest_loc=lowest_loc, lowest_energy=lowest_energy);

            # update final output (display all workers as stopped as well):
            if cnt_workers>0 and self.logging:
                if mainnet and debugging==False:
                    clear_output(wait=True);
                if mainnet:
                    LOC_MIN, ENERGY_MIN, MALLOB_CHIPS, details = _get_status_details_api(JOB_ID, annealing_time, all_stopped = True);
                else:
                    LOC_MIN, ENERGY_MIN, MALLOB_CHIPS = 0,0,0;
                    details = "";
                elapsed_time = time.process_time() - t;
                if mainnet:
                    # display:
                    table = ([['DYNEXJOB','BLOCK FEE','ELAPSED','WORKERS READ','CHIPS','STEPS','GROUND STATE']]);
                    table.append([JOB_ID, price_per_block, elapsed_time, cnt_workers, total_chips, total_steps, lowest_energy]);
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
                # format: xxx.dnx.32.1.0.0.000000
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
                    for var in range(0, self.num_variables-8): # REMOVE VALIDATION VARS
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
                print("[DYNEX] SAMPLESET READY");
            
            # create return sampleset: ------------------------------------------------------------------------------------------------------
            sampleset_clean = [];
            for sample in sampleset:
                sample_dict = _Convert(sample);
                sampleset_clean.append(sample_dict);

        except KeyboardInterrupt:
            if mainnet:
                _cancel_job_api(JOB_ID, self.logging);
            print("[DYNEX] Keyboard interrupt");
            return {'error': 'Keyboard interrupt'};

        except Exception as e:
            if mainnet:
                _cancel_job_api(JOB_ID, self.logging);
            print("[DYNEX] Exception encountered:", e);
            return {'error':'Exception encountered', 'details':e};

        self.sampleset = sampleset_clean;

        return self.dimod_assignments; 
        
