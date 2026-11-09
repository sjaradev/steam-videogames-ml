"""Pruebas reproducibles del prototipo local con modelos preentrenados."""
from fastapi.testclient import TestClient
from api.main import app

def test_health_and_profiles():
 with TestClient(app) as c:
  assert c.get('/health').json()['model_loaded'] is True
  assert len(c.get('/profiles').json()) == 12
  assert len(c.get('/tags').json()['tags']) == 150

def test_prediction_and_validation():
 with TestClient(app) as c:
  response=c.post('/predict',json={'tags':['Action','FPS','Shooter']})
  assert response.status_code == 200
  payload=response.json()
  assert 0<=payload['cluster']<12
  assert payload['distance_to_centroid']>=0
  assert c.post('/predict',json={'tags':['NOT_IN_VOCABULARY']}).status_code == 422
  assert c.post('/predict',json={'tags':[]}).status_code == 422

def test_search_interface_and_monitor():
 with TestClient(app) as c:
  assert c.get('/games?query=Counter-Strike').status_code==200
  assert c.get('/ui').status_code==200
  assert c.get('/map.png').status_code==200
  assert c.get('/metrics').status_code==200
  assert c.get('/monitor').status_code==200

def test_search_ranking_normalization_and_pagination():
 with TestClient(app) as c:
  response=c.get('/games',params={'query':'  COUNTER strike  ','limit':1})
  assert response.status_code==200
  first=response.json()
  assert first['results'][0]['name']=='Counter-Strike'
  assert first['has_more'] is True
  second=c.get('/games',params={'query':'counter-strike','limit':1,'offset':1}).json()
  assert second['matches']==first['matches']
  assert second['results'][0]['appid']!=first['results'][0]['appid']
  assert c.get('/games',params={'query':'   '}).json()['results']==[]
  assert c.get('/games',params={'query':'***'}).json()['results']==[]
  hidden=c.get('/games',params={'query':'love','hide_adult':True}).json()
  assert all(not g['adult_content'] for g in hidden['results'])
