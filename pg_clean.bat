docker compose -f docker-compose-pg.yaml down -v

docker image prune -a

docker volume prune -a

docker ps -a

docker images

echo.
echo ========================================
:pause_end
echo.
pause