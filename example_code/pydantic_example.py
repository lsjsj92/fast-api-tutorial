from typing import List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, BaseSettings
from pydantic import Field
from pydantic import validator


class DBConfig(BaseSettings):
    host: str = Field(default='127.0.0.1', env='db_host')
    port: int = Field(default=3306, env='db_port')
    
    # 먼저 실행
    @validator("host", pre=True)
    def check_host(cls, host_input):
        if host_input == 'localhost':
            return "127.0.0.1"
        return host_input
    
    # validator error를 설정할 수 있음
    @validator("port")
    def check_port(cls, port_input):
        if port_input not in [3306, 8080]:
            raise ValueError("port error")
        return port_input

class ProjectConfig(BaseModel):
    project_name: str = 'soojin'
    db_info: DBConfig = DBConfig()
    
    

data = {
    'project_name': '이수진의 프로젝트',
    'db_info': {
        'host': 'localhost',
        'port' : 3306
    }
}

my_pjt = ProjectConfig(**data)
print(my_pjt.dict())
print(my_pjt.db_info)

