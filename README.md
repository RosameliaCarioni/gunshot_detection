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
docker run -it -v /Users/rosameliacarioni/University/Thesis/code:/tf/notebooks -p 8888:8888 gunshot
```
pwd to find path 

To attach to a container that is running, grab the name as above and run:
```
docker exec -it gunshot bash
```