# Bachelor thesis on Gunshot Identification
## Student: Rosamelia Carioni 
## Supervisor: Pietro Bonizzi 

To get all the required packages do:
pip install -r requirements.txt


## Docker Commands
To list images
```
docker image ls
```

To see running containers
```
docker ps -a
```

To run a container, find the name using ps -a and then 
```
docker run -p 8888:8888 -it -v () [CONTAINER-NAME]
```

To attach to a container that is running, grab the name as above and run:
```
docker exec -it [CONTAINER-NAME] bash
```