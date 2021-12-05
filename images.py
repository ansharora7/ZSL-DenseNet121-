from arguments import  parse_args
args = parse_args()
from dataset import NIHChestXray
NIHChestXray(args, args.test_file, transform=None, classes_to_load='all')