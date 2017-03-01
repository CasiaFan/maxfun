#! /bin/bash
pidof cs224d.sh 
while [ $? -ne 0 ]
do
	echo "Process exit abnormally!"
	sh cs224d.sh
done	
echo "Process done!"
:<<'END'
case "$(pidof cs224d.sh | wc -w)" in
0)	echo "Process down! Restarting!"
	sh cs224d.sh &
	;;
1)	echo "Process is running!"
	;;
*)	echo "Multiple processes! Kill first one."
	kill $(pidof cs224d.sh | awk '{print $1}')
	;;
esac
END

