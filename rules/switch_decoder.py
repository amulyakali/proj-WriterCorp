import requests
import sys, random
import json, ast
from scipy.spatial import distance

URL  = "http://ec2-52-7-45-213.compute-1.amazonaws.com:8103/icr_res"

switch_kws = [ 'SETTE', 'REJ', 'PURG', 'REMAIN', 'DISPEN', 'TOTAL' ]
counter_kws = ['C1','C2','C3','C4','C5','INC','IC','OU','OC','DEC', 'DC','END','EC', 'ST']

counter_start = [ 'C1','C2', 'ST' ]

def findStart( rowD, slipType ):
	if slipType == 'COUNTER':
		for key, val in rowD.items():
			rowText = ''
			for _item in val:
				rowText += ' ' + _item['text']
			rowText = rowText.strip()	
			for kw in counter_start:
				if kw in rowText: return key
	
	return None


def txtIsKey( st ):
	for kk in counter_kws:
		if kk in st: return True

	return False 

def txtisAmount( st ):
	num = ''	
	st = st.replace('C1','').replace('C2','').replace('C3','').replace('C4','').replace('C5','')
	if 'XXXX' in st or ':' in st: return False
	if len( st.split('/') ) > 2: return False ## rule out dates
	for char in st:
		if ord(char) >= 48 and ord(char) <= 57: num += char
	
	if len(st) == len(num) and len(num) > 2: return True		
	if len(st) >=4 and len(num) > 2: return True
	if 'RS' in st and len(st) > 3 and 'RS' not in st[-3: ]: return True ## dont return TRUE if the contour is just RS. or RS,
	if st == 'O' or st == '0': return True
	if 'RS.0' in st: return True
	if 'RS0' in st: return True
	if 'RSO' in st : return True
	if '0 C' in st : return True
	return False 

def getNum( txt ):
	nn = ''
	for char in txt:
		if ord(char) >= 48 and ord(char) <= 57: nn += char
		else: break
	return nn

def emptyOutLL( ll ):
	end_ , out_ = False, False
	for elem in ll:
		if 'END' in elem: end_ = True 
		if 'EC' in elem and not 'DEC' in elem: end_ = True 
		if 'OU' in elem or 'OC' in elem: out_ = True 
	
	if end_ and out_ : return []
	return ll

def doesNotContain( text, ll ):

	for elem in ll:
		if elem in text: return False
	return True

def retMin( rowLL, x, y, txtVal ,minHt, ll ):

	minDist, minElem = 100000, None
	minD_ = dict()
	gg = []
	for elem in rowLL:
			'''
			if txtIsKey( elem['text'] ) is False or\
				elem['pts'][0] > x or\
				abs(y - elem['pts'][1]) > minHt : continue

			print('CHEKOV - ', elem['text'] , txtisAmount( elem['text'] ) )
			dist_ = distance.euclidean( ( x, y ), ( elem['pts'][0], elem['pts'][1] ) )
			if dist_ > 0:
				minD_[ int( dist_ ) ] = elem	
			if dist_ < minDist and dist_ > 0 and doesNotContain( elem['text'] , ll ):
				minDist = dist_
				minElem = elem
			'''
			if txtIsKey( elem['text'] ) is False or ( elem['pts'][1] > y and (abs(y - elem['pts'][1]) > minHt) ) or y > elem['pts'][-1]: continue 
			gg.append( elem )
			minD_[ abs( elem['pts'][1] - y ) ] = elem

	st = list( minD_.keys() )
	st.sort()
	if len(st) > 0:	
		gg = st
		## first check for ST, IC etv
		print('BEGIN CHECK FOR retMin - ', gg )
		for cc in (gg):
			kk = minD_[ cc ]
			for ik in counter_kws[5:]:
				print('chotu ckey, rowl',ik, kk['text'] )
				tt = kk['text']
				if ik in tt or tt in ik:
					print('chot nan maga -', tt, returnKeyName( tt ) )
					kname = returnKeyName( tt )
					if kname not in ll or kname == 'STR': 
						print('RETMIN - FIRST', txtVal, kk )
						return kk	
		## now check c1, c3 etc
		for cc in (gg):
			kk = minD_[ cc ]
			for ik in counter_kws[:5]:
				tt = kk['text']
				if ik in tt:
					kname = returnKeyName( tt )
					if kname not in ll: 
						print('RETMIN - 2nd', txtVal, kk )
						return kk	

	
	'''	
	if len( minD_ ) > 1:
		ll_ = (list(minD_.keys()))
		ll_.sort()
		print('ROKKAM - MIN 1', minD_[ ll_[0] ], ' MIN 2', minD_[ ll_[1] ], y ,ll )
		print('MORE - ', abs( minD_[ ll_[0] ]['pts'][1] - y ) <= minHt,\
				abs( minD_[ ll_[1] ]['pts'][1] - y ) <= minHt,\
				abs( minD_[ ll_[1] ]['pts'][1] - y ) <= abs( minD_[ ll_[0] ]['pts'][1] - y ) ) 
		## if both #1 and 2 are within Y minHt range send the closest Y
		if abs( minD_[ ll_[0] ]['pts'][1] - y ) <= minHt and\
		   abs( minD_[ ll_[1] ]['pts'][1] - y ) <= minHt and\
		abs( minD_[ ll_[1] ]['pts'][1] - y ) <= abs( minD_[ ll_[0] ]['pts'][1] - y ):
			#doesNotContain( minD_[ ll_[1] ]['text'] , ll ):
			#ll.append( minD_[ ll_[1] ]['text'] ) 
			if minD_[ ll_[1] ]['text'] in counter_kws[:5]: return minD_[ ll_[0] ], ll
			elif minD_[ ll_[0] ]['text'] in counter_kws[:5]: return minD_[ ll_[1] ], ll
			elif abs( minD_[ ll_[1] ]['pts'][1] - minD_[ ll_[0] ]['pts'][1] ) < minHt\
				and abs( minD_[ ll_[0] ]['pts'][0] - x ) < abs( minD_[ ll_[1] ]['pts'][0] - x ):
				return minD_[ ll_[0] ], ll
			elif abs( minD_[ ll_[1] ]['pts'][1] - minD_[ ll_[0] ]['pts'][1] ) < minHt\
				and abs( minD_[ ll_[0] ]['pts'][0] - x ) > abs( minD_[ ll_[1] ]['pts'][0] - x ):
				return minD_[ ll_[1] ], ll
			return minD_[ ll_[1] ], ll
		elif abs( minD_[ ll_[0] ]['pts'][1] - y ) <= minHt and\
		   abs( minD_[ ll_[1] ]['pts'][1] - y ) <= minHt and\
		abs( minD_[ ll_[1] ]['pts'][1] - y ) == abs( minD_[ ll_[0] ]['pts'][1] - y ) and\
			len( ll ) == 0 and 'ST' in minD_[ ll_[1] ]['text']:
			#ll.append( minD_[ ll_[1] ]['text'] ) 
			return minD_[ ll_[1] ], ll
	'''	
				
	#if minElem is not None:	
	#	ll.append( minElem['text'] )
	return minElem

def findClosest( rowLL, x, y, txtVal ,minHt, ll ):

	#ll = emptyOutLL( ll )
	
	if 'ENDRS' in txtVal:# or 'STRRS' in txtVal:
		_arr = txtVal.split('ENDRS')
		txtVal = _arr[0]+' ENDRS '+_arr[1]
	
	if 'STRRS' in txtVal:# or 'STRRS' in txtVal:
		_arr = txtVal.split('STRRS')
		txtVal = _arr[0]+' STRRS '+_arr[1]
			
	minDist, minElem = 100000, None
	print('---FINDING CLOSEST ', txtVal, ll )
	## first try
	txt_ = txtVal.split()
	if minElem is None and len(txt_) > 1:
	## look out for missing keys and text like 1000DEC RS. 28888	
		last_idx = 0	
		
		for ctr in range(len(txt_)):
			## extract number
			kk = txtisAmount( txt_[ctr] )
			if kk is False: continue
			print('UGUY -', kk, txt_[ctr] )
			## if ctr is 0 meaning the very first word is an amount then we will need to hunt
			## for key in the prev contours..eg 12345 END RS. 1234 ..here 1234 will be atrributed
			## to END but 12345 needs a parent	
			if ctr == 0:
				minElem = retMin( rowLL, x, y, txt_[ctr] ,minHt, ll )
				print('BOZO - ',minElem, txt_[ctr])
				if minElem is not None:
					minElem = { 'text_0': minElem['text'], 'pts_0': minElem['pts'] }
					last_idx += 1
				elif minElem is None:
					minElem = { 'text_0': 'UNK', 'pts_0': 'NA' }
					last_idx += 1
					
			for inner in range(ctr):
				print( txtIsKey( txt_[ inner ] ), txt_[ inner ] )
				if txtIsKey( txt_[ inner ] ): 
					if minElem is not None:
						val_ = list(minElem.values())
						if txt_[ inner ] not in val_ and ( last_idx > 0 and\
							minElem[ 'pts_'+str(last_idx-1) ] != 'CURR' )  :
							minElem[ 'text_'+str(last_idx) ] = txt_[ inner ]
							minElem[ 'pts_'+str(last_idx) ] = 'CURR'
							last_idx += 1
						elif txt_[ inner ] not in val_ and last_idx == 0: 
							minElem[ 'text_'+str(last_idx) ] = txt_[ inner ]
							minElem[ 'pts_'+str(last_idx) ] = 'CURR'
							last_idx+= 1
						elif txt_[ inner ] not in val_ and \
							minElem[ 'pts_'+str(last_idx-1) ] == 'CURR' :
							minElem[ 'text_'+str(last_idx-1) ] += txt_[ inner ]
							#ll.append( txt_[ inner ] )
					else:
						minElem = { 'text_'+str(last_idx): txt_[ inner ], \
								'pts_'+str(last_idx): 'CURR' } 
						last_idx+= 1
						#ll.append( txt_[ inner ] )

	## 2nd attempt using standard params	
	if minElem is None:
		minElem = retMin( rowLL, x, y, txtVal ,minHt, ll )

	# if still not found increase minHt
	## because of folds in the images there could either be a +ve or a -ve diff
	## between prev elem Y and current elem Y ..this also extends to distance
	if minElem is None:
		minHt += 5	
		minElem = retMin( rowLL, x, y, txtVal ,minHt, ll )

	return minElem, ll	

def returnNumericalValue( txt ):
	## filters out C1, C2 etc .since they also might end up in final numbers
	txt = txt.replace('C1','').replace('C2','').replace('C3','').replace('C4','').replace('C5','')
	_arr = txt.split()
	numArr = []
	for word in _arr:
		print('COMING into returnNumericalValue - ', word , txtisAmount( word ))
		if txtisAmount( word ) is True:
			wd = word.replace('/','7').replace('I','1').replace('O','0')
			num = ''
			print('Final wd ', wd)
			if wd[-3:] == '.00' and 'RS0' not in wd and not 'RSO' in wd:
				wd = wd[:-3]
			elif wd[-2:] == '.0' and 'RS.0' not in wd:
				wd = wd[:-2]
			for char in wd:
				if ord(char) >= 48 and ord(char) <= 57: num += char
				#if '.' == char: num += char
			if num != '':
				numArr.append( int(float(num)) )

	return numArr

def returnKeyName( txt ):
	if 'ST' in txt: return 'STR'
	if 'INC' in txt or 'IC' in txt: return 'INC'
	if 'OU' in txt or 'OC' in txt: return 'OUT'
	if 'DEC' in txt or 'DC' in txt: return 'DEC'
	if 'END' in txt or ( 'EC' in txt and not 'DEC' in txt ): return 'END'

def checkIfMultipleKeys( keyL ):
	newLL = []
	for txt in keyL:
		if ( 'INC' in txt or 'IC' in txt ) and ( 'DEC' in txt or 'DC' in txt ):
			newLL.append( 'INC' )
			newLL.append( 'DEC' )
		elif ( 'OU' in txt or 'OC' in txt ) and ( 'END' in txt or ( 'EC' in txt and not 'DEC' in txt ) ):
			newLL.append( 'OUT' )
			newLL.append( 'END' )
		elif ( 'OU' in txt or 'OC' in txt ) and ( 'DEC' in txt ):
			newLL.append( 'OUT1' )
			newLL.append( 'DEC' )
		elif ( 'INC' in txt or 'IC' in txt ) and ( 'ST' in txt ):
			newLL.append( 'ST' )
			newLL.append( 'INC' )
		else: newLL.append( txt )
	if len( newLL ) > len( keyL ): return newLL, True
	else:
		return newLL, False

def returnDKey( finalReturnDict, curr_elem ):
	cx, cy = curr_elem['pts'][:2]
	closest_dist, counter = 100000, -1
	for key, val in finalReturnDict.items():
		if 'STR' not in val: continue
		cmpx , cmpy = val['STR']['value_co_ords'][2], val['STR']['value_co_ords'][1] 
		dist_ = abs( cy - cmpy )
		if dist_ < closest_dist:
			closest_dist = dist_
			counter = key
	return closest_dist, counter

def returnKeyRowNum( lookup_, assigned, curr_x,  str_x, curr_y, str_y, str_h ):
		key_loc, row_num = 'NA', 'NA'
		if curr_x > str_x and abs( curr_y - str_y ) < str_h: ## meaning its in same row as STR
			## can only mean its INC ad format is STR - INC ; OU - DEC ; OU - END
			key_loc = 'INC'
			row_num = 1	
		elif curr_x <= str_x and abs( curr_y - str_y ) >= str_h and \
			abs( curr_y - str_y ) < 2*str_h :
			if 'INC' not in assigned:
			## meaning its in the NEXT row as STR
			## can only mean its INC ad format is STR ; INC - DEC ; OU - END
				key_loc = 'INC'
				row_num = 2	
			else:
				key_loc = 'OUT1' ## other format which has 2 OUs
				row_num = 2	
		elif curr_x > str_x and abs( curr_y - str_y ) >= str_h and abs( curr_y - str_y ) < 2*str_h: 
			## meaning its in the NEXT row as STR
			## can only mean its DEC ad format is STR ; INC - DEC ; OU - END
			key_loc = 'DEC'
			row_num = 2	
		elif curr_x <= str_x and abs( curr_y - str_y ) >= 2*str_h: 
			## meaning its in the 3rd row as STR
			## can only mean its OUT ad format is STR ; INC - DEC ; OU - END
			if 'INC' in assigned and lookup_['INC']['row_num'] == 2:
				key_loc = 'OUT'
				row_num = 3	
			elif 'INC' in assigned and lookup_['INC']['row_num'] == 1: # other format which has 2 OU
				key_loc = 'OUT2'
				row_num = 3	
		elif curr_x > str_x and abs( curr_y - str_y ) >= 2*str_h: 
			## meaning its in the 3rd row as STR
			## can only mean its END ad format is STR ; INC - DEC ; OU - END
			key_loc = 'END'
			row_num = 3	
		
		return key_loc, row_num	

def finalDecryption( ll, curr_elem, key, finalReturnDict , minHt ):
	## if key has no "pts" element then it means it is self referential
	## so we can use co-ordinates from "curr_elem" instead of "key"

	#storeNumericalVals[ str(len(storeNumericalVals) ) ] = curr_elem		
	## check if key has ST/C1/C2 
	for key_, val_ in key.items():
		if 'text' in key_ and ( 'ST' in val_ )\
			and ( 'OUT' in ll or 'END' in ll  ):
			print('UNFILLED PREV COUNTER BUT RESETTING LL')
			ll = []
			break 	
			#if 'text' in key_ and ( 'ST' in val_ or val_ in ['C2','C3','C4','C5'] )\
	lastDKey = ''
	if len(ll) == 0 and finalReturnDict == {}:
		DKey = 'Counter#0'
		finalReturnDict[ DKey ] = {}	
	elif len(ll) == 0 and finalReturnDict != {}:
		key_ll = list(finalReturnDict.keys())
		key_ll.sort()
		lastKey = int(key_ll[-1].split('#')[-1])	
		lastDKey = 'Counter#'+str(lastKey)
		DKey = 'Counter#'+str(lastKey+1)
		finalReturnDict[ DKey ] = {}	
	else:
		key_ll = list(finalReturnDict.keys())
		key_ll.sort()
		DKey = key_ll[-1]
		print( 'CLOSEST STR = ', returnDKey( finalReturnDict, curr_elem ) )

	curr_keys_ll =list( finalReturnDict[ DKey ].keys() )
	base_y = None	
	base_y1 = None	
	if 'OUT' in curr_keys_ll :
		base_y = finalReturnDict[ DKey ]['OUT']['value_co_ords'][1]
	elif 'OUT2' in curr_keys_ll :
		base_y1 = finalReturnDict[ DKey ]['OUT2']['value_co_ords'][1]
	elif 'END' in curr_keys_ll :
		base_y = finalReturnDict[ DKey ]['END']['value_co_ords'][1]

	fmtNrml = True	
	if 'STR' in curr_keys_ll and 'INC' in curr_keys_ll:
		if ( finalReturnDict[ DKey ]['STR']['value_co_ords'][1] - \
			finalReturnDict[ DKey ]['INC']['value_co_ords'][1] ) <= minHt:
			fmtNrml = False

	if base_y is not None and abs( base_y - curr_elem['pts'][1] ) > minHt and fmtNrml:
		ll = []
		key_ll = list(finalReturnDict.keys())
		key_ll.sort()
		lastKey = int(key_ll[-1].split('#')[-1])	
		DKey = 'Counter#'+str(lastKey+1)
		finalReturnDict[ DKey ] = {}	
		if 'text' in key: key['text'] = 'STR'
		print('BALGOPAL')

	if base_y1 is not None and abs( base_y1 - curr_elem['pts'][1] ) > minHt and fmtNrml is False:
		ll = []
		key_ll = list(finalReturnDict.keys())
		key_ll.sort()
		lastKey = int(key_ll[-1].split('#')[-1])	
		DKey = 'Counter#'+str(lastKey+1)
		finalReturnDict[ DKey ] = {}	
		if 'text' in key: key['text'] = 'STR'
	
	print('000000000000 ENTERING finalDecryption for ', curr_elem, key, ll, finalReturnDict, DKey )	

	if len(ll) == 0 and 'pts' in list(key.keys()) and 'ST' in key['text']: 
		## meaning this is the regular key with co-ords
		locD = finalReturnDict[ DKey ]
		ll.append( 'STR' )
		locD['STR'] = { 'value': returnNumericalValue( curr_elem['text'] )[0], \
					'value_co_ords': curr_elem['pts'], \
					'key_co_ords': key['pts'], \
					'row_num': 1 }	
		print('xxxxxxxxxxxxxxx finalDecryption IF 1', locD)
	elif len(ll) == 0 and 'pts' in list(key.keys()) and \
	( 'C1' in key['text'] or 'C2' in key['text'] or 'C3' in key['text'] or 'C4' in key['text'] ):  
		## meaning this is the regular key with co-ords
		locD = finalReturnDict[ DKey ]
		ll.append( 'STR' )
		locD['STR'] = { 'value': returnNumericalValue( curr_elem['text'] )[0], \
					'value_co_ords': curr_elem['pts'], \
					'key_co_ords': key['pts'], \
					'row_num': 1 }	
		print('xxxxxxxxxxxxxxx finalDecryption IF 2', locD)
	elif len(ll) == 0 and 'pts' not in list(key.keys()) and 'text_0' in list(key.keys()) \
		and key['text_0'] in counter_kws[:5]:
			if type( key['pts_0'] ) is str and key['pts_0'] == 'CURR':
				kc = curr_elem['pts']
			else:
				kc = key['pts_0']
			locD = finalReturnDict[ DKey ]
			ll.append( 'STR' )
			locD['STR'] = { 'value': returnNumericalValue( curr_elem['text'] )[0], \
					'value_co_ords': curr_elem['pts'], \
					'key_co_ords': kc, \
					'row_num': 1 }	

	elif len(ll) > 0 and 'pts' in list(key.keys()) and\
	( 'ST' in key['text'] or 'EC' in key['text'] or 'OC' in key['text'] or 'OU' in key['text'] \
	  or 'DC' in key['text'] or 'IC' in key['text']  or 'IN' in key['text'] or 'END' in key['text']):  
		
		key_loc = returnKeyName( key['text'] )
		assigned = list( finalReturnDict[ DKey ].keys() )
		## in case end / out comes after new row due to tilt, assigne to prev DKey
		## it should only be checked in case the next rows INC / DEC haven't yet been assigned
		## if it has then there's no chance the OCR has actually read that value
		if 'INC' not in assigned and 'DEC' not in assigned and \
			( 'OC' in key['text'] or 'OUT' in key['text'] or 'OU' in key['text'] or  \
				 ( 'EC' in key['text'] and not 'DEC' in key['text'] ) or 'END' in key['text'] )\
			and int(DKey.split('#')[-1]) > 0:
			idx = int(DKey.split('#')[-1]) - 1
			old_ll = list(finalReturnDict[ 'Counter#'+str(idx) ].keys())
			if ( ( ( 'OC' in key['text'] or 'OUT' in key['text'] ) and\
				'OUT' not in old_ll ) or ( ( 'EC' in key['text'] or 'END' in key['text'] ) and\
								 'END' not in old_ll  ) ) and\
				'STR' in finalReturnDict[ 'Counter#'+str(idx) ]:
				base_ = finalReturnDict[ 'Counter#'+str(idx) ]['STR']
				lookup_ = finalReturnDict[ 'Counter#'+str(idx) ]	
			else:
				return finalReturnDict, ll 	
		numVal = returnNumericalValue( curr_elem['text'] )[0]
		curr_x, curr_y = curr_elem['pts'][:2] ## compare key co-ords to STR val co-ords

		row_num = 'NA'
		## for INC only
		
		if key_loc not in assigned:
			ll.append( key_loc )
			finalReturnDict[ DKey ][key_loc] = { 'value': numVal, \
					'value_co_ords': curr_elem['pts'], \
					'key_co_ords': key['pts'], \
					'row_num': row_num }	
			print('xxxxxxxxxxxxxxx finalDecryption IF 3 NORMAL KEYS ', finalReturnDict[ DKey ])

	elif len(ll) > 0 and 'pts' in list(key.keys()) and \
	( 'C1' in key['text'] or 'C2' in key['text'] or 'C3' in key['text'] \
		or 'C4' in key['text']  or 'C5' in key['text'] or 'UNK' in key['text']):  

		assigned = list( finalReturnDict[ DKey ].keys() )
		## in case end / out comes after new row due to tilt, assigne to prev DKey
		## it should only be checked in case the next rows INC / DEC haven't yet been assigned
		## if it has then there's no chance the OCR has actually read that value
		if 'INC' not in assigned and 'DEC' not in assigned and \
			( 'OC' in key['text'] or 'OUT' in key['text'] or \
				 'EC' in key['text'] or 'END' in key['text'] ):
			idx = int(DKey.split('#')[-1]) - 1
			base_ = finalReturnDict[ 'Counter#'+str(idx) ]['STR']
			lookup_ = finalReturnDict[ 'Counter#'+str(idx) ]	
		else:
			base_ = finalReturnDict[ DKey ]['STR']
			lookup_ = finalReturnDict[ DKey ]	
		numVal = returnNumericalValue( curr_elem['text'] )[0]
		curr_x, curr_y = curr_elem['pts'][:2] ## compare key co-ords to STR val co-ords
		str_x, str_y, str_h = base_[ 'value_co_ords' ][2], base_[ 'value_co_ords' ][1], minHt 
		
		print('xxxxxxxxxxxxxxx finalDecryption IF 4 ABNORMAL KEYS curr_x curr_y str_x str_y str_h '\
				,curr_x, curr_y,str_x, str_y, str_h )
		## for INC only
		key_loc = 'UNK'
		row_num = 'NA'
		if curr_x > str_x and abs( curr_y - str_y ) < str_h: ## meaning its in same row as STR
			## can only mean its INC ad format is STR - INC ; OU - DEC ; OU - END
			key_loc = 'INC'
			row_num = 1	
		elif curr_x <= str_x and abs( curr_y - str_y ) >= str_h and \
			abs( curr_y - str_y ) < 2*str_h :
			if 'INC' not in assigned :
			## meaning its in the NEXT row as STR
			## can only mean its INC ad format is STR ; INC - DEC ; OU - END
				key_loc = 'INC'
				row_num = 2	
			elif 'INC' in assigned and lookup_['INC']['row_num'] != 'NA':
				key_loc = 'OUT1' ## other format which has 2 OUs
				row_num = 2	
		elif curr_x > str_x and abs( curr_y - str_y ) >= str_h and abs( curr_y - str_y ) < 2*str_h: 
			## meaning its in the NEXT row as STR
			## can only mean its DEC ad format is STR ; INC - DEC ; OU - END
			key_loc = 'DEC'
			row_num = 2	
		elif curr_x <= str_x and abs( curr_y - str_y ) >= 2*str_h: 
			## meaning its in the 3rd row as STR
			## can only mean its OUT ad format is STR ; INC - DEC ; OU - END
			if 'INC' in assigned and lookup_['INC']['row_num'] == 2:
				key_loc = 'OUT'
				row_num = 3	
			elif 'INC' in assigned and lookup_['INC']['row_num'] == 1: # other format which has 2 OU
				key_loc = 'OUT2'
				row_num = 3	
		elif curr_x > str_x and abs( curr_y - str_y ) >= 2*str_h: 
			## meaning its in the 3rd row as STR
			## can only mean its END ad format is STR ; INC - DEC ; OU - END
			key_loc = 'END'
			row_num = 3	
		if key_loc not in lookup_:
			ll.append( key_loc )	
			lookup_[key_loc] = { 'value': numVal, \
					'value_co_ords': curr_elem['pts'], \
					'key_co_ords': key['pts'], \
					'row_num': row_num }	
			print('xxxxxxxxxxxxxxx finalDecryption IF 4 Variables ', curr_x, str_x, curr_y, str_y, str_h)
			print('xxxxxxxxxxxxxxx finalDecryption IF 4 AB-NORMAL KEYS ', lookup_)

	elif 'pts' not in list(key.keys()):
		ignore = True
		if 'STR' in finalReturnDict[ DKey ]:
			ignore = False
			assigned = list( finalReturnDict[ DKey ].keys() )
			## in case end / out comes after new row due to tilt, assigne to prev DKey
			## it should only be checked in case the next rows INC / DEC haven't yet been assigned
			## if it has then there's no chance the OCR has actually read that value
			'''
			if 'INC' not in assigned and 'DEC' not in assigned and \
				( 'OC' in key['text'] or 'OUT' in key['text'] or \
					 'EC' in key['text'] or 'END' in key['text'] ):
				idx = int(DKey.split('#')[-1]) - 1
				base_ = finalReturnDict[ 'Counter#'+str(idx) ]['STR']
				lookup_ = finalReturnDict[ 'Counter#'+str(idx) ]	
			else:
			'''
			if 1 == 1:
				base_ = finalReturnDict[ DKey ]['STR']
				lookup_ = finalReturnDict[ DKey ]	
			numVal = returnNumericalValue( curr_elem['text'] )[0]
			curr_x, curr_y = int( curr_elem['pts'][0] ),int( curr_elem['pts'][1] )
			str_x, str_y, str_h = base_[ 'value_co_ords' ][2], base_[ 'value_co_ords' ][1], minHt 
		## self referential - C3 STRRS. 1857000
		keyL = []
		rejectD = {}
		## at max u ll get 2 genuine keys .. filter out C1, C2 etc
		for key1, val in key.items():
			if 'pts' in key1: 
				continue
			if ( txtIsKey( val ) or val == 'UNK' ) and val not in counter_kws[:5]: 
				keyL.append( val )
			elif txtIsKey( val ) and val in counter_kws[:5]: 
				co = key[ 'pts_'+ key1.split('_')[-1] ]
				if type(co) is str and co == 'CURR': rejectD[ val ] = curr_elem['pts']
				else: rejectD[ val ] = co

		if len(keyL) == 0 and len( rejectD ) > 0:	
			formatNormal = True
			print('PICKING KEY - ', curr_x, curr_y, str_x, str_y, str_h )
			if 'INC' in assigned and lookup_['INC']['row_num'] == 1:
				formatNormal = False
			for kk, vv in rejectD.items():
				if curr_x > str_x and abs( curr_y - str_y ) <= str_h: 
					keyL.append( 'INC' )
					formatNormal = False
				elif curr_x <= str_x and abs( curr_y - str_y ) >= str_h and\
					 abs( curr_y - str_y ) < 2*str_h: 
					keyL.append( 'INC' )
				elif curr_x > str_x and abs( curr_y - str_y ) >= str_h and\
					abs( curr_y - str_y ) < 2*str_h:
					keyL.append( 'DEC' )
				elif curr_x <= str_x and abs( curr_y - str_y ) >= str_h and \
				abs( curr_y - str_y ) < 2*str_h and formatNormal is False:
					keyL.append( 'OUT1' )
				elif curr_x <= str_x and \
				abs( curr_y - str_y ) >= 2*str_h and formatNormal is False:
					keyL.append( 'OUT2' )
				elif curr_x <= str_x and \
				abs( curr_y - str_y ) >= 2*str_h and formatNormal is True:
					keyL.append( 'OUT' )
				elif curr_x > str_x and abs( curr_y - str_y ) >= 2*str_h:
					keyL.append( 'END' )

		numVals = returnNumericalValue( curr_elem['text'] )
		print( 'Checking MULTIUPLE KEYS B4 and AFt,', keyL )
		keyL, multiCheck = checkIfMultipleKeys( keyL )
		print( 'Checking MULTIUPLE KEYS B4 and AFt,', keyL )
		locD = finalReturnDict[ DKey ]
		print('xxxxxxxxxxxxxxx finalDecryption IF 5 numVals, keyL ', numVals, keyL)
		if len( numVals ) != len( keyL ):
			print('NOT ADDING ANYTHING SINCE SZ OF NUMVAL AND KEYL DIFFER')
		else:
			for ctr in range(len(numVals)):
				if keyL[ctr] == 'UNK': continue

				key_loc = returnKeyName( keyL[ctr] )
				if key_loc is None:
					key_loc = 'UNK_'+str( random.randint(10, 30) )
				
				if 'pts_'+str(ctr) in key and key[ 'pts_'+str(ctr) ] != 'CURR':
					key_coords = key[ 'pts_'+str(ctr) ]
				else:
					key_coords = curr_elem['pts']
				#curr_x, curr_y = int(key_coords[0]), int(key_coords[1]) ## compare key co-ords to STR val co-ords
				## if the value is 1234 END RS 1456, 1456 will have CURR as co-ords
				## but 1234 will have the prev key as proper 4 pt co-ord , hence
				## the below checks if co-ord is an STR simply use the current text co-ords
				
				row_num = 'NA'
				if ignore is False:		
					if key_loc == 'INC' and curr_x > str_x and abs( curr_y - str_y ) <= str_h: 
						row_num = 1
					elif key_loc == 'INC' and curr_x <= str_x and abs( curr_y - str_y ) >= str_h and \
                                        abs( curr_y - str_y ) < 2*str_h : 
						row_num = 2
					if key_loc == 'DEC': row_num = 2
					if key_loc == 'OUT' and curr_x <= str_x and abs( curr_y - str_y ) >= str_h and \
					abs( curr_y - str_y ) < 2*str_h :
						key_loc = 'OUT1'
						row_num = 2
					elif key_loc == 'OUT' and curr_x <= str_x and \
					abs( curr_y - str_y ) >= 2*str_h :
						key_loc = 'OUT2'
						row_num = 3
					elif key_loc == 'END': row_num = 3
				elif ignore is True and key_loc == 'STR':
					row_num = 1
				'''	
				elif ignore is True and key_loc != 'STR':
					key_loc = 'STR'
					row_num = 1
				'''	
				print(' ROWNUM check = ', ignore, row_num )
				if ignore is False and row_num == 'NA' and key_loc!= 'STR': # confirm location
					if curr_x > str_x and abs( curr_y - str_y ) <= str_h: 
						key_loc == 'INC'
						row_num = 1
					elif curr_x <= str_x and abs( curr_y - str_y ) >= str_h: 
						key_loc == 'INC'
						row_num = 2
					if curr_x > str_x and abs( curr_y - str_y ) >= str_h and \
                                        abs( curr_y - str_y ) < 2*str_h: 
						key_loc == 'DEC'
						row_num = 2
					if curr_x <= str_x and abs( curr_y - str_y ) >= str_h and \
					abs( curr_y - str_y ) < 2*str_h :
						key_loc = 'OUT1'
						row_num = 2
					elif curr_x <= str_x and \
					abs( curr_y - str_y ) >= 2*str_h :
						key_loc = 'OUT2'
						row_num = 3
					elif curr_x > str_x and \
					abs( curr_y - str_y ) >= 2*str_h :
						key_loc = 'END'
						row_num = 3
				
				if len(ll) == 0 and 'OUT' in key_loc and lastDKey != '':
				
					lastCtr = finalReturnDict[ lastDKey ]		
					if 'END' in lastCtr and 'OUT' not in lastCtr and \
						'OUT1' not in lastCtr and 'OUT2' not in lastCtr:
						print('BUMBLE BEE ',lastCtr)
						refy = lastCtr['END']['value_co_ords'][1]
						curry = curr_elem['pts'][1]
						if abs( refy - curry ) < abs( curr_elem['pts'][-1] \
										- curr_elem['pts'][1] ):
							print('OUT BELONGS TO PREV CTR !!!')
							ll.append( key_loc )	
							lastCtr[key_loc] = { 'value': numVals[ ctr ], \
								'value_co_ords': curr_elem['pts'], \
								'key_co_ords': key_coords, \
								'row_num': row_num, 'multi_key': multiCheck }	
				elif 'STR' in key_loc:
					key_ll = list(finalReturnDict.keys())
					key_ll.sort()
					lastKey = int(key_ll[-1].split('#')[-1])	
					DKey = 'Counter#'+str(lastKey+1)
					interim = {}	
					ll.append( key_loc )	
					interim[key_loc] = { 'value': numVals[ ctr ], \
						'value_co_ords': curr_elem['pts'], \
						'key_co_ords': key_coords, \
						'row_num': row_num, 'multi_key': multiCheck }	
					finalReturnDict[ DKey ] = interim
				elif key_loc not in locD:
					ll.append( key_loc )	
					locD[key_loc] = { 'value': numVals[ ctr ], \
						'value_co_ords': curr_elem['pts'], \
						'key_co_ords': key_coords, \
						'row_num': row_num, 'multi_key': multiCheck }	

		print('xxxxxxxxxxxxxxx finalDecryption IF 5 CONJOINED KEYS ', locD)
	
	finalKeyL = list( finalReturnDict[ DKey ].keys() )
	ll = finalKeyL	
	if 'END' in finalKeyL:# and ( 'OUT' in finalKeyL or 'OUT1' in finalKeyL or 'OUT2' in finalKeyL ):
		ll = []

	return finalReturnDict, ll

def findClosestSTR( finalReturnDict, cx, cy, val ):
	dist_ = {}
	curr_txt = val['text']
	curr_pts = val['pts']
	curr_h = val['pts'][-1] - val['pts'][1]
	keyLL = checkIfMultipleKeys( [curr_txt] )
	numLL = returnNumericalValue( curr_txt )

	if len( numLL ) == 0: return finalReturnDict

	multiKeyCheck = False

	if len( keyLL ) > 1 and len( numLL ) > 1 and len( keyLL ) == len( numLL ):
		multiKeyCheck = True

	for key, val in finalReturnDict.items():
		for ikey, ival in val.items():
			if ikey == 'STR' and ( cy > ival['value_co_ords'][1] or ( ival['value_co_ords'][1] > cy and ival['value_co_ords'][1] - cy <= curr_h ) ):#and abs( cy - ival['value_co_ords'][1] ) < curr_h:
				dd = abs(  cy - ival['value_co_ords'][1] )
				dist_[ dd ] = {'pts': ival['value_co_ords'], 'text': key }

	if len( dist_ ) == 0: # meaning, current contour Y is trying to look for STR even below itself
		return finalReturnDict

	ll_ = list(dist_.keys())
	ll_.sort()
	ref_ctr = dist_[ ll_[0] ]
	refD = finalReturnDict[ ref_ctr['text'] ]
	ref_x, ref_y, ref_h = ref_ctr['pts'][0], ref_ctr['pts'][1],\
		min( ( ref_ctr['pts'][-1] - ref_ctr['pts'][1] )	, ( curr_pts[-1] - curr_pts[1] ) ) - 2
	# above we take a min since at times STR height is wierdly 30-40% higher than other KEY hiehgts
	print('For contour ', curr_txt , ' at X, Y ', cx, cy, curr_pts, ' closest STR is frm ', ref_ctr['text'] )	
	print('CECKING IN ', ref_x, ref_y, ref_h, ' from DD ', refD['STR']['value_co_ords'], refD.keys() )

	formatNrml = True
	if ( 'INC' in refD and refD['INC'] != {} and abs( refD['INC']['value_co_ords'][1] - ref_y ) < ref_h ) or\
	( cx != ref_x and cy != ref_y and abs( cy  - ref_y ) <= ref_h and cx <= ref_x ):
		formatNrml = False

	if multiKeyCheck is False:
		if ( 'INC' not in refD or ( 'INC' in refD and refD['INC'] == {} ) ) and not formatNrml and \
			abs( cy  - ref_y ) <= ref_h and cx > ref_x :
			## INC abnormal format
			finalReturnDict[ ref_ctr['text'] ]['INC'] = {'value': numLL[0],\
								'value_co_ords': curr_pts,\
								'row_num': 1 }

		elif ( 'INC' not in refD or  ( 'INC' in refD and refD['INC'] == {} ) ) and formatNrml and \
			abs( cy  - ref_y ) > ref_h and abs( cy  - ref_y ) <= 2*ref_h and cx <= ref_x :
			## INC normal format
			finalReturnDict[ ref_ctr['text'] ]['INC'] = {'value': numLL[0],\
								'value_co_ords': curr_pts,\
								'row_num': 2 }

		elif ( 'OUT' not in refD or ( 'OUT' in refD and refD['OUT'] == {} ) ) and formatNrml and \
			abs( cy  - ref_y ) >= 2*ref_h and cx <= ref_x :
			## OUT normal format
			## in case END present check if OUT Y1 is minHt > END Y1 
			## coz in some cases, if STR for next counter not found
			## stuff from next STR creep into current one
			if 'END' in finalReturnDict[ ref_ctr['text'] ] and \
			finalReturnDict[ ref_ctr['text'] ]['END'] != {} and\
			abs( finalReturnDict[ ref_ctr['text'] ]['END']['value_co_ords'][1] - \
				cy ) < ref_h:
				finalReturnDict[ ref_ctr['text'] ]['OUT'] = {'value': numLL[0],\
								'value_co_ords': curr_pts,\
								'row_num': 3 }

		elif ( 'OUT1' not in refD or ( 'OUT1' in refD and refD['OUT1'] == {} ) ) and not formatNrml and \
			abs( cy  - ref_y ) > ref_h and abs( cy  - ref_y ) <= 2*ref_h and cx <= ref_x :
			## OUT normal format
			finalReturnDict[ ref_ctr['text'] ]['OUT1'] = {'value': numLL[0],\
								'value_co_ords': curr_pts,\
								'row_num': 2 }

		elif ( 'OUT2' not in refD or ( 'OUT2' in refD and refD['OUT2'] == {} ) ) and not formatNrml and \
			abs( cy  - ref_y ) > 2*ref_h and cx <= ref_x :
			## OUT normal format
			finalReturnDict[ ref_ctr['text'] ]['OUT2'] = {'value': numLL[0],\
								'value_co_ords': curr_pts,\
								'row_num': 3 }

		elif ( 'DEC' not in refD or ( 'DEC' in refD and refD['DEC'] == {} ) ) and \
			abs( cy  - ref_y ) > ref_h and abs( cy  - ref_y ) < 2*ref_h and cx > ref_x :
			## OUT normal format
			finalReturnDict[ ref_ctr['text'] ]['DEC'] = {'value': numLL[0],\
								'value_co_ords': curr_pts,\
								'row_num': 2 }

		elif ( 'END' not in refD or ( 'END' in refD and refD['END'] == {} ) ) and \
			abs( cy  - ref_y ) > 2*ref_h and cx > ref_x :
			## OUT normal format
			finalReturnDict[ ref_ctr['text'] ]['END'] = {'value': numLL[0],\
								'value_co_ords': curr_pts,\
								'row_num': 3 }
	return finalReturnDict		

def prefinal( finalReturnDict, storeNumericalVals, counter_bounds ):
	counter_keys = list(counter_bounds.keys())
	counter_keys.sort()
	finalVals = dict()

	for key, val in storeNumericalVals.items():
		cx, cy = val['pts'][:2]
		numArr = returnNumericalValue( val['text'] )
		if len( numArr ) > 1 : continue
		dkey = None
		prev_ub, prev_lb, prev_key = None, None, None
		bm_ub, bm_lb, str_marker = None, None, False
		for ik, ival in counter_bounds.items():
			ub, lb = ival[0], ival[1]
			if (cy > ub and cy <= lb): 
				dkey = ik
				bm_ub, bm_lb = ub, lb
				break
			if ( prev_ub is not None and cy < ub and cy > prev_lb ): 
				dkey = ik
				str_marker = True
				bm_ub, bm_lb = ub, lb
				break
			prev_ub, prev_lb, prev_key = ub, lb, ik
		print( 'prefinal: For item ', val['text'], ' assigned counter is ', dkey, bm_ub, bm_lb  )
		assign_ = None
		if dkey is not None:
			locD = finalReturnDict[ dkey ]
			avail_keys = list(locD.keys())	
			if 'STR' in avail_keys:
				base_x, base_y = locD['STR']['value_co_ords'][:2]
				print('Goin to eval ', cx, cy, base_x, base_y )
				if cx < base_x: # candidates can be INC/OUT/1/2
					if abs( base_y - cy ) < 0.5*( bm_lb - bm_ub ):
						print('prefinal: For item ', val['text'], ' in counter ', \
							dkey, ' assigned key INC') 
						assign_ = 'INC'	
					elif 'INC' in avail_keys and \
					abs( base_y - cy ) < 0.5*( bm_lb - bm_ub ) and\
					abs( locD['INC']['value_co_ords'][1] - base_y ) < 10 and \
					'OUT1' not in avail_keys and 'OUT2' not in avail_keys and \
					'OUT' not in avail_keys:
						print('prefinal: For item ', val['text'], ' in counter ', \
							dkey, ' assigned key OUT1') 
						assign_ = 'OUT1'	
					elif 'INC' in avail_keys and \
					abs( base_y - cy ) > 0.5*( bm_lb - bm_ub ) and\
					abs( locD['INC']['value_co_ords'][1] - base_y ) < 10 and \
					'OUT1' not in avail_keys and 'OUT2' not in avail_keys and \
					'OUT' not in avail_keys:
						print('prefinal: For item ', val['text'], ' in counter ', \
							dkey, ' assigned key OUT2') 
						assign_ = 'OUT2'	
					elif 'INC' in avail_keys and \
					abs( base_y - cy ) > 0.5*( bm_lb - bm_ub ) and\
					abs( locD['INC']['value_co_ords'][1] - base_y ) > 10 and \
					abs( locD['INC']['value_co_ords'][1] - base_y ) < bm_lb and \
					'OUT1' not in avail_keys and 'OUT2' not in avail_keys and \
					'OUT' not in avail_keys:
						print('prefinal: For item ', val['text'], ' in counter ', \
							dkey, ' assigned key OUT') 
						assign_ = 'OUT'	
				if cx >= base_x: # candidates can be INC/DEC/END
					if abs( base_y - cy ) < 10 :
						print('prefinal: For item ', val['text'], ' in counter ', \
							dkey, ' assigned key INC') 
						assign_ = 'INC'	
					elif abs( base_y - cy ) <= 0.5*( bm_lb - bm_ub ): 
						print('prefinal: For item ', val['text'], ' in counter ', \
							dkey, ' assigned key DEC') 
						assign_ = 'DEC'	
					elif abs( base_y - cy ) > 0.5*( bm_lb - bm_ub ): 
						print('prefinal: For item ', val['text'], ' in counter ', \
							dkey, ' assigned key END') 
						assign_ = 'END'	
			elif 'STR' not in avail_keys and str_marker:
				print('prefinal: For item ', val['text'], ' in counter ', \
                                                        dkey, ' assigned key STR')
				assign_ = 'STR'	
			if dkey in finalVals:
				prn_dd = finalVals[ dkey ]
			else:
				prn_dd = dict()
			
			numLL = returnNumericalValue( val['text'] )
			if len( numLL ) == 0:
				numLL = [0] # last ditch option
			prn_dd[ assign_ ] = {'value': numLL[0],\
								'value_co_ords': val['pts'],\
								'row_num': 'NA' }
			finalVals[ dkey ] = prn_dd
	
	print('BOMBER -- ', finalVals )	
	for key, val in finalVals.items():
		finale = finalReturnDict[ key ]
		ll_ = list(finale.keys())
		for ik, iv in val.items():
			if ik not in ll_:
				finale[ ik ] = iv
	print('PRAYER - ', finalReturnDict )

def returnCtrLoc( finalReturnDict, ctr_key ):
	ll_1 = list( finalReturnDict.keys() )
	ll_1.sort()
	idx = 0
	for ele in ll_1:
		if len( finalReturnDict[ ele ] ) == 0: continue
		if ctr_key == ele: return idx
		idx += 1

def triangulate( finalReturnDict, storeNumericalVals, counter_bounds, storeCounterVals ):
	mm , adder = None, None

	for key, val in storeCounterVals.items():
		cval, cy = val['text'], val['pts'][1]
		for ctr_key, bounds in counter_bounds.items():
			if cy >= bounds[0] and cy < bounds[1]:
				print('Counter from slip is #',cval,' and FD key is ',ctr_key)
				if 'C1' in cval: mm = 1		
				elif 'C2' in cval: mm = 2
				elif 'C3' in cval: mm = 3
				elif 'C4' in cval: mm = 4
				elif 'C5' in cval: mm = 5 
				elif 'C6' in cval: mm = 6 
				ckey = returnCtrLoc( finalReturnDict, ctr_key )
				if ckey is not None and mm > ckey:	
					adder = mm - ckey
				break
		if adder is not None: 
			idx = 0
			for key, val in finalReturnDict.items():
				if len( val ) == 0: continue ## ignore empty dicts	
				val['CTR'] = 'C'+str( idx + adder )
				idx += 1
			break
	print('GUZMAN - ', finalReturnDict )
	## FINAL TRIANG LEG
	for key, val in finalReturnDict.items():	
		if len( val ) == 0: continue

		triangError = False
		str_, inc_, dec_, out_, end_ = 0, 0 , 0, 0, 0
		if 'STR' in val: str_ = int(val['STR']['value']) 	
		elif 'STR'  not in val: 
			str_ = 0
			val['STR'] = { 'value': 0, 'value_co_ords': 'NA' }
		if 'INC' in val: inc_ = int(val['INC']['value']) 	
		elif 'INC'  not in val: 
			inc_ = 0
			val['INC'] = { 'value': 0, 'value_co_ords': 'NA' }
		if 'DEC' in val: dec_ = int(val['DEC']['value']) 	
		elif 'DEC'  not in val:
			dec_ = 0
			val['DEC'] = { 'value': 0, 'value_co_ords': 'NA' }
		if 'OUT' in val: out_ = int(val['OUT']['value']) 	
		elif 'OUT1' in val: out_ = int(val['OUT1']['value']) 	
		elif 'OUT2' in val: out_ = int(val['OUT2']['value']) 	
		elif 'OUT2' not in val and 'OUT1' not in val and 'OUT' not in val:
			out_ = 0
			val['OUT'] = { 'value': 0, 'value_co_ords': 'NA' }
		if 'END' in val: end_ = int(val['END']['value']) 	
		elif 'END'  not in val:
			end_ = 0
			val['END'] = { 'value': 0, 'value_co_ords': 'NA' }
		

		score_keeper = {}
		print('B4 CALC str_ inc_  ( out_  dec_ )  end_', str_ , inc_ , out_ , dec_ , end_ )
		if str_ + inc_ - ( out_ + dec_ ) != end_:
			val['TRIANGULATION'] = 'FAIL'
		else:
			val['TRIANGULATION'] = 'PASS'

	delkey = []
	for key, val in finalReturnDict.items():	
		if 'CTR' not in val:
			print('FINAL DELETION - ',key, ' SINCE ITS MOSTLY EMPTY' )
			delkey.append(key)
		if None in val: del finalReturnDict[ key ][None]

	for elem in delkey: del finalReturnDict[ elem ]
	'''
		## STR + INC - ( OUT + DEC ) = END
		if str_ + inc_ - ( out_ + dec_ ) != end_:  	
			triangError = True
			## first keep str const	
			if str_ != 0:
				calc_str_ = end_ + ( out_ + dec_ ) - inc_
				print( '1 :', calc_str_, str_ )
				if abs(calc_str_) > str_:
					score_keeper[ 'STR' ] = abs(calc_str_) / str_
				else:
					score_keeper[ 'STR' ] = str_ /abs(calc_str_) 
			if inc_ != 0:
				calc_inc_ = end_ + ( out_ + dec_ ) - str_
				print( '2 :', calc_inc_, inc_ )
				if abs(calc_inc_) > inc_:
					score_keeper[ 'INC' ] = abs(calc_inc_) / inc_
				else:
					score_keeper[ 'INC' ] = (inc_) / abs(calc_inc_)
			if out_ != 0:
				calc_out_ = end_ + ( dec_ ) - ( inc_ + str_ )
				print( '3 :', calc_out_, out_ )
				if abs(calc_out_) > out_:
					score_keeper[ 'OUT' ] = abs(calc_out_) / out_
				else:
					score_keeper[ 'OUT' ] = out_/ abs(calc_out_)
			if dec_ != 0:
				calc_dec_ = end_ + ( out_ ) - ( inc_ + str_ )
				print( '4 :', calc_dec_, dec_ )
				if abs(calc_dec_) > dec_:
					score_keeper[ 'DEC' ] = abs(calc_dec_) / dec_
				else:
					score_keeper[ 'DEC' ] = dec_ / abs(calc_dec_) 
			if end_ != 0:
				calc_end_ = ( inc_ + str_ ) - ( out_ + dec_ )
				print( '5 :', calc_end_, end_ )
				if abs(calc_end_) > end_:
					score_keeper[ 'END' ] = abs(calc_end_) / end_
				else:
					score_keeper[ 'END' ] = end_ / abs(calc_end_) 
			print('SCORE KEEPERRRRR - ', score_keeper )
			sorted_ = sorted( score_keeper.items(), key=lambda x: x[1] )
			print('SCORE KEEPERRRRR SORTED TOP TO LOW- ', score_keeper )
			err_key = list( score_keeper.keys() )[-1]
			print('AFTER TRIANG ERROR KEY = ', err_key ) 
			if err_key == 'STR': 
				print('TRIANGULATION FIX: STR FROM', val[ err_key ]['value'], ' TO ', \
						end_ + ( out_ + dec_ ) - inc_ )
				val[ err_key ]['value'] = end_ + ( out_ + dec_ ) - inc_	
			if err_key == 'INC': 
				print('TRIANGULATION FIX: INC FROM', val[ err_key ]['value'], ' TO ', \
						end_ + ( out_ + dec_ ) - str_ )
				val[ err_key ]['value'] = end_ + ( out_ + dec_ ) - str_	
			if err_key == 'DEC': 
				print('TRIANGULATION FIX: DEC FROM', val[ err_key ]['value'], ' TO ', \
						end_ + ( out_ ) - ( str_ + inc_ ) )
				val[ err_key ]['value'] = end_ + ( out_ ) - ( str_ + inc_ )
			if err_key == 'OUT': 
				print('TRIANGULATION FIX: OUT FROM', val[ err_key ]['value'], ' TO ', \
						end_ + ( dec_ ) - ( str_ + inc_ ) )
				val[ err_key ]['value'] = end_ + ( dec_ ) - ( str_ + inc_ )
			if err_key == 'END': 
				print('TRIANGULATION FIX: END FROM', val[ err_key ]['value'], ' TO ', \
						( str_ + inc_ ) - ( out_ + dec_ ) )
				val[ err_key ]['value'] =  ( str_ + inc_ ) - ( out_ + dec_ )
	print('GUZMAN2 - ', finalReturnDict )
	'''

def beginXform( _ll , fname ):
	rowD = dict()
	prevY = None
	minY = 100000
	maxY = -100000
	text_ll = []

	# slipType = decideSlipType( text_ll )
	storeNumericalVals = dict()
	storeCounterVals = dict()
	minHt = 15
	masterLL = []
	for dict_ll in _ll:
		for dict_ in dict_ll:
			loc = dict_['text']
			print('Goin in - ', loc )	
			if 'STRRS' in loc:
				kk = loc.split('STRRS')
				dict_['text'] = kk[0]+' STR RS'+' '.join( kk[1:] )
				loc = dict_['text']
			if 'STRS' in loc:
				kk = loc.split('STRS')
				dict_['text'] = kk[0]+' ST RS'+' '.join( kk[1:] )
				loc = dict_['text']
			if 'S1RS' in loc:
				kk = loc.split('S1RS')
				dict_['text'] = kk[0]+' ST RS'+' '.join( kk[1:] )
				loc = dict_['text']
			if 'SIRS' in loc:
				kk = loc.split('SIRS')
				dict_['text'] = kk[0]+' ST RS'+' '.join( kk[1:] )
				loc = dict_['text']
			if 'IR RS' in loc:
				kk = loc.split('IR RS')
				dict_['text'] = kk[0]+' ST RS'+' '.join( kk[1:] )
				loc = dict_['text']
			if 'OUTRS' in loc:
				kk = loc.split('OUTRS')
				dict_['text'] = kk[0]+' OUT RS'+' '.join( kk[1:] )
				loc = dict_['text']
			if 'OURS' in loc:
				kk = loc.split('OURS')
				dict_['text'] = kk[0]+' OU RS'+' '.join( kk[1:] )
				loc = dict_['text']
			if 'ENDRS' in loc:
				kk = loc.split('ENDRS')
				dict_['text'] = kk[0]+' END RS'+' '.join( kk[1:] )
				loc = dict_['text']
			if 'RS' in loc:
				numd = 0
				for ch in loc:
					if ord(ch) >= 48 and ord(ch) <= 57: numd += 1
				if numd >= len(loc)-3:
					dict_['text'] = loc.replace(' ','')
				loc = dict_['text']
			if 'RS. ' in loc:
				_arr = loc.split('RS. ')
				numd = 0
				for ch in _arr[-1]:
					if ord(ch) >= 48 and ord(ch) <= 57: numd += 1
				if numd >= 4:
					dict_['text'] = _arr[0]+'RS.'+' '.join( _arr[1:] )
				loc = dict_['text']
			if 'RS.0' in loc:
				ll = loc.split( 'RS.0' )
				if len(ll) > 1:
					dict_['text'] = ll[0]+' RS.0 '+' '.join( ll[1:] )
				loc = dict_['text']
			elif 'RS:' in loc:
				ll = loc.split( 'RS:' )
				if len(ll) > 1:
					dict_['text'] = ll[0]+' RS '+' '.join( ll[1:] )
				loc = dict_['text']
			elif 'RS.D' in loc:
				ll = loc.split( 'RS.D' )
				if len(ll) > 1:
					dict_['text'] = ll[0]+' RS.0 '+' '.join( ll[1:] )
				loc = dict_['text']
			elif 'RS. 0' in loc:
				ll = loc.split( 'RS. 0' )
				if len(ll) > 1:
					dict_['text'] = ll[0]+' RS.0 '+' '.join( ll[1:] )
				loc = dict_['text']
			elif 'RS.O' in loc:
				ll = loc.split( 'RS.O' )
				if len(ll) > 1:
					dict_['text'] = ll[0]+' RS.0 '+' '.join( ll[1:] )
				loc = dict_['text']
			elif 'RS. O' in loc:
				ll = loc.split( 'RS. O' )
				if len(ll) > 1:
					dict_['text'] = ll[0]+' RS.0 '+' '.join( ll[1:] )
				loc = dict_['text']
			elif 'RS,0' in loc:
				ll = loc.split( 'RS,0' )
				if len(ll) > 1:
					dict_['text'] = ll[0]+' RS.0 '+' '.join( ll[1:] )
				loc = dict_['text']
			elif 'RS,O' in loc:
				ll = loc.split( 'RS,O' )
				if len(ll) > 1:
					dict_['text'] = ll[0]+' RS.0 '+' '.join( ll[1:] )
				loc = dict_['text']
			masterLL.append( dict_ )
			print('Goin out - ', loc )	

	## sort rowD elements using x co-ords
	neoD = {}	
	assignedKeys = []	
	finalReturnDict = {}
	beginRead = False	
	sender = masterLL.copy()

	beginRead = False	
	for ctr in range(len(sender)):
			elem = sender[ ctr ]
			locT =  elem['text']
			print( 'BEGIN TEST JOURNEY - ',locT, beginRead )
			if beginRead is False and\
			 ( 'C1' in locT or 'C2' in locT or 'C3' in locT or 'C4' in locT or 'ST' in locT ):
				beginRead = True
			elif beginRead is False:
				continue 
			if 'C1' in locT or 'C2' in locT or 'C3' in locT \
				or 'C4' in locT or 'C5' in locT or 'C6' in locT:
				storeCounterVals[ str(len(storeCounterVals)) ] = elem

			if txtisAmount( elem['text'] ) is False: continue
			print( 'BEGIN JOURNEY - ',locT, beginRead )
			close_, assignedKeys = findClosest( sender[ ctr-3: ctr ], elem['pts'][0], elem['pts'][1], \
								elem['text'] ,minHt, assignedKeys )
			print( 'X Y  ', elem['text'] , elem['pts'][0], elem['pts'][1], \
			int( elem['pts'][1] )*100+int(elem['pts'][0]), ' CLOSEST KEY = ', close_ )
			if close_ is None: continue
			finalReturnDict, assignedKeys = finalDecryption( assignedKeys, elem , \
								close_,finalReturnDict, minHt )

	##final pass 

	for elem in masterLL: print( elem ) 		
	print('--------------------- B4')
	counter_bounds = {}
	for key, val in finalReturnDict.items():
		if len(val) == 0: continue
		start_y , end_y = [], []
		print( '\n',key,'\n=================' )
		for k, v in val.items():
			start_y.append( v['value_co_ords'][1] ) 
			end_y.append( v['value_co_ords'][1] ) 
			print(k,' : ', v )	
		start_y.sort()
		end_y.sort()
		counter_bounds[ key ] = ( start_y[0], end_y[-1] ) ## min start y and max end y

	print('COUNTER BOUNDS = ', counter_bounds )
	for ctr in range(len(masterLL)):
			elem1 = sender[ ctr ]
			locT =  elem1['text']
			print( 'TEST - ',locT, beginRead )
			if beginRead is False and\
			 ( 'C1' in locT or 'C2' in locT or 'C3' in locT or 'C4' in locT or 'ST' in locT ):
				beginRead = True
			elif beginRead is False:
				continue 
			print('MOT - ', txtisAmount( elem1['text'] ) )
			if txtisAmount( elem1['text'] ) is False: continue
			storeNumericalVals[ str(len(storeNumericalVals) ) ] = elem1

	print( storeNumericalVals )	
	prefinal( finalReturnDict, storeNumericalVals, counter_bounds )
	triangulate( finalReturnDict, storeNumericalVals, counter_bounds, storeCounterVals )

	print('--------------------- AFTER')
	for key, val in finalReturnDict.items():
		if len(val) == 0: continue
		print( '\n',key,'\n=================' )
		for k, v in val.items():
			print(k,' : ', v )	

	return finalReturnDict
	'''
	print('YEEEEPPPPPPPPPPPPPPPPPPP')
	for key, val in storeNumericalVals.items():
		print(key, val)
	'''
		
def call_api(path):
	files = {'file': open(path, 'rb')}
	op = requests.post(URL,files=files)
	kk = ast.literal_eval ( op.json() )
	_ll = kk["lines"]
	fname = (path.split('/')[-1])	
	resp = beginXform( _ll, fname )
	print( resp )

if __name__=="__main__":
    path = sys.argv[1]
    call_api(path)

