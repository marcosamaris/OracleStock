 cat bovespa.csv|while read line
  do
    read -d, sticker < <(echo $lineprint("checking if any null values are present\n", df_ge.isna().sum())
    curl 'https://query1.finance.yahoo.com/v7/finance/download/'$sticker'?period1=946689180&period2=1590196339&interval=1d&events=history&crumb=XXXXXXX' -H 'user-agent: Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0)' -H 'cookie: B=YYYYYY;' -H 'referer: https://finance.yahoo.com/quote/'$sticker'/history?p='$sticker >> ./data/$sticker.csv
    
  done
sticker=CSAN3.SA
   # rsync -CrazpP 'https://query1.finance.yahoo.com/v7/finance/download/'$sticker'?period1=946689180&period2=1590196339&interval=1d&events=history&crumb=XXXXXXX'  ./

    curl 'https://query1.finance.yahoo.com/v7/finance/download/'$sticker'?period1=946689180&period2=1590196339&interval=1d&events=history&crumb=XXXXXXX' -H 'user-agent: Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0)' -H 'cookie: B=YYYYYY;' -H 'referer: https://finance.yahoo.com/quote/'$sticker'/history?p='$sticker >> ./data/$sticker.csv
