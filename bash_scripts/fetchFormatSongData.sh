#!/bin/bash
let BEARER

get_new_bearer() {
	BEARER=`curl -s -X "POST" -H "Authorization: Basic YzJjOTg0ZmQ0OGYxNGM4ZTlkNzhkYWFjOWZhMjYxZWE6NjE2MTg5NzBjZjgyNGZlZWE2NGYxYmRiMTk2OWQxZTM=" -d grant_type=client_credentials https://accounts.spotify.com/api/token | sed 's/^.*"access_token":"//' | sed 's/","token_type.*$//'`
}

let TMP_FILE
let CSV_FILE
TMP_TMP_FILE="err_tmp_song.txt"
TMP_FILE="tmp_song.txt"
#TMP_FILE="errorOut"
CSV_FILE="songs"$1".csv"

let CURR_URI
let ERR_OUT
let TOO_MANY_OUT
let RATE_LIMIT

echo -n '' | cat  > $CSV_FILE

get_song() {
	while true
	do
		get_song_data1
		ERR_OUT=`cat $TMP_TMP_FILE | grep -c '"error":'`
		TOO_MANY_OUT=`cat $TMP_TMP_FILE | grep  -c 'Too many requests'`
		RATE_LIMIT=`cat $TMP_TMP_FILE | grep  -c '"message": "API rate limit exceeded"'`
		if ((ERR_OUT))
		then
			echo "ERROR"
			get_new_bearer
		fi
		if (($((TOO_MANY_OUT+RATE_LIMIT))))
		then
			echo "Rate Limit Exceeded, Sleeping..."
			sleep 30
			echo "Done sleep"
		fi
		if ! ((ERR_OUT))
		then
			if ! ((TOO_MANY_OUT))
			then
				break
			fi
		fi
	done
	cat $TMP_TMP_FILE > $TMP_FILE
	while true
	do
		get_song_data2
		ERR_OUT=`cat $TMP_TMP_FILE | grep -c '"error":'`
		TOO_MANY_OUT=`cat $TMP_TMP_FILE | grep  -c 'Too many requests'`
		RATE_LIMIT=`cat $TMP_TMP_FILE | grep  -c '"message": "API rate limit exceeded"'`
		if ((ERR_OUT))
		then
			echo "ERROR"
			get_new_bearer
		fi
		if (($((TOO_MANY_OUT+RATE_LIMIT))))
		then
			echo "Rate Limit Exceeded, Sleeping..."
			sleep 30
			echo "Done sleep"
		fi
		if ! ((ERR_OUT))
		then
			if ! ((TOO_MANY_OUT))
			then
				break
			fi
		fi
	done
	cat $TMP_TMP_FILE >> $TMP_FILE
}

get_song_data1() {
	curl -s --request GET --url 'https://api.spotify.com/v1/audio-features/'$CURR_URI --header 'Authorization: Bearer '$BEARER --header 'Content-Type: application/json' > $TMP_TMP_FILE
}

get_song_data2() {
	curl -s --request GET --url 'https://api.spotify.com/v1/tracks/'$CURR_URI --header 'Authorization: Bearer '$BEARER --header 'Content-Type: application/json' > $TMP_TMP_FILE
}

format_song_data() {
	cat $TMP_FILE | sed -r 's/^\s*\{\s*$|^\s*\}\s*$|^\s*\},\s*$|^\s*\}\{\s*$|^.*"external_urls" :.*$|^.*"href" :.*$|^.*"height" :.*$|^.*"width" :.*$|^.*"url" :.*$|^.*"track_href" :.*$|^.*"analysis_url" :.*$|^.*"uri" :.*$|^.*"spotify" :.*$|^.*"total_tracks" :.*$|^.*"available_markets" :.*$|^.*"disc_number" :.*$|^.*"is_local" :.*$|^.*"track_number" :.*$|^.*"preview_url" :.*$|^.*"type" :.*$|^.*"release_date_precision" :.*$|^.*"album_type" :.*$|^.*"isrc" :.*$|^.*"artists" :.*$|^.*"album" :.*$|^.*"external_ids" :.*$|^.*"explicit" :.*$|^.*"images" :.*$|^\s*\}, \{\s*$|^\s*\} \],\s*$//' | awk '/name/{c+=1}{if(c==1){sub("name","artist_name",$0)}else if(c==2){sub("name","album_name",$0)}else if(c==4){sub("name","song_name",$0)};print}' | awk '/id/{c+=1}{if(c==1){sub("id","song_id",$0)};print}' | sed -r 's/^.*"name" :.*$|^.*"id" :.*$//' | sed -r '/^\s*$/d' | sed 's/^\s*//' | sed 's/\(^"time_signature" :.*$\)/\1,/' | sed 's/^".*" : //' | sed 's/^"\(.*\)",/\1,/' | tr -d ',' | sed 's/$/,/' | tr '\n' ' ' | sed 's/, $//' >> $CSV_FILE
	echo '' >> $CSV_FILE
}

get_new_bearer

let i=0

while read -r CURR_URI
do
	get_song
	format_song_data
	((i=i+1))
	if ! ((i%100))
	then
		get_new_bearer
		echo $i
	fi
done < "uris"