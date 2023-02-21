import sys

import uvicorn

from packages import APIConfig
from packages import DataHandler
from packages import ProjectConfig


class FastAPIRunner(ProjectConfig, DataHandler):
    def __init__(self, args):        
        self.host = args.host
        self.port = args.port
        DataHandler.__init__(self)

    def run(self):
        api_info_data = {
            'api_info': {
                'host': self.host,
                'port' : self.port,
            }
        }
        # API config data type 체크 
        api_info_data = self.check_type(APIConfig, api_info_data)
        uvicorn.run(f"{api_info_data.api_name}", host=api_info_data.api_info.host, port=api_info_data.api_info.port, reload=True)
    
    
