import os
import re

def matchTLS(input_file, target_dir, target='.tif'):
    """ This is a convenience function to match two files linked by TLS ID.

    :param input_file: Filename with the TLS ID to be matched.
    :type input_file: str
    :param target_dir: Filepath to the directory to look for matches.
    :type: target_dir: str
    :param target: The file ending for valid inputs, defaults to '.tif'
    :type target: str

    :return: Returns the full filepath to the matching file, or 'no match found.'
    :rtype: str
    """
    
    match = extractTLSID(input_file)
    for i in sorted(os.listdir(target_dir)):
        if not i.endswith(target):
            continue
        tls_id = extractTLSID(i)
        if tls_id == match:
            return(target_dir + '/' + i)
    return ('no match found')

def extractTLSID(input_str):
    """ Convenience function to extract TLS ID. 

    :param input_str: String containing TLS ID.
    :type input_str: str    

    :return: Returns the extracted TLS ID.
    :rtype: str
    """
    tls_id = re.match("TLS_\d{4}_\d{8}_\d{2}", input_str)
    if tls_id:
        return(tls_id.group())
