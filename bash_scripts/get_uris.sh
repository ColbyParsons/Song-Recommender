#!/bin/bash
grep 'track_uri' "mpd.slice.0"*".json" | sort | uniq | sed 's/^.*spotify:track:\(.*\)",/\1/' > '../../all_uri1'
for i in {1..9} 
do
	touch "../../all_uri$i"
	for j in {0..9}
	do
		for k in {0..9}
		do
			echo "$i$j$k"
			grep 'track_uri' "mpd.slice.$i$j$k"*".json" | sort | uniq | sed 's/^.*spotify:track:\(.*\)",/\1/' > '../../temp_uri'
			cat "../../all_uri$i" >> '../../temp_uri'
			cat '../../temp_uri' | sort | uniq > "../../all_uri$i"
		done
	done
done