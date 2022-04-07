def get_config(name, args=None):
    if name == 'nr':
        from .opt_NR import Config
    elif name == 'aid':
        from .opt_AID import Config
    elif name == 'rsd46':
        from .opt_RSD46 import Config
    elif name == 'ucm':
        from .opt_UCMerced import Config
    elif name == 'pnt':
        from .opt_PatternNet import Config
    elif name == 'tg2rgb':
        from .opt_TianGong2_RGB import Config
    elif name == 'eurorgb':
        from .opt_EuroSAT_RGB import Config

    if args is not None:
        mOptions = Config(args)
    else:
        mOptions = Config()

    return mOptions

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR+'/Tools/dltoos')
