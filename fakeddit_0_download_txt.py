"""
Only if needed. Complete data can be found at ../fakeddit
"""

import gdown

id = '1DuH0YaEox08ZwzZDpRMOaFpMCeRyxiEF'
gdown.download_folder(id=id, quiet=False, use_cookies=False)

id = '1hjcLrwMOGJBHrB6bqlaIhn6N09X0lr85'
gdown.download(id=id, output="multimodal_only_samples/all_comments.tsv.zip", quiet=False)