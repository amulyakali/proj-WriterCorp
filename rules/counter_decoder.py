import requests
import sys, random
import json, ast
from scipy.spatial import distance

URL  = "http://ec2-52-7-45-213.compute-1.amazonaws.com:8103/icr_res"

switch_kws = [ 'SETTE', 'REJ', 'PURG', 'REMAIN', 'DISPEN', 'TOTAL' ]
counter_kws = ['C1','C2','C3','C4','C5','INC','IC','OU','OC','DEC', 'DC','END','EC', 'ST']

counter_start = [ 'C1','C2', 'ST' ]

def _counter_findStart( rowD, slipType ):
	if slipType == 'COUNTER':
		for key, val in rowD.items():
			rowText = ''
			for _item in val:
				rowText += ' ' + _item['text']
			rowText = rowText.strip()	
			for kw in counter_start:
				if kw in rowText: return key
	
	return None

def _counter_decideSlipType( body ):
	## at least 2 kws for switch and 3 for counter
	sw_ctr , ctr_ctr = set(), set()
	for row in body:
		for sw in switch_kws:
			if sw in row: sw_ctr.add( sw )
		for ct in counter_kws:
			if ct in row: ctr_ctr.add( ct )

	if len( sw_ctr ) >= 2: return 'SWITCH'	
	if len( ctr_ctr ) >= 3: return 'COUNTER'
	return 'UNK'	

def _counter_txtIsKey( st ):
	for kk in switch_kws:
		if kk.upper() in st.upper(): return True

	return False 

def _counter_txtisAmount( st ):
	num = ''	
	if 'XXXX' in st or ':' in st: return False
	if len( st.split('/') ) > 2: return False ## rule out dates
	for char in st:
		if ord(char) >= 48 and ord(char) <= 57: num += char
	
	if len(st) == len(num) and len(num) > 2: return True		
	if len(st) >=4 and len(num) > 3: return True
	return False 

def _counter_getNum( txt ):
	nn = ''
	for char in txt:
		if ord(char) >= 48 and ord(char) <= 57: nn += char
		else: break
	return nn

def _counter_doesNotContain( text, ll ):

	for elem in ll:
		if elem in text: return False
	return True

def _counter_retMin( rowLL, x, y, txtVal ,minHt, ll ):

	minDist, minElem = 100000, None
	minD_ = dict()
	gg = []
	for elem in rowLL:
			if _counter_txtIsKey( elem['text'] ) is False : continue 
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
			for ik in switch_kws:
				print('chotu ckey, rowl',ik, kk['text'] )
				tt = kk['text']
				if ik.upper() in tt.upper() or tt.upper() in ik.upper():
					print('chot nan maga -', tt, _counter_returnKeyName( tt ) )
					kname = _counter_returnKeyName( tt )
					if 1 == 1: 
					#if kname not in ll or 'SETTE' in kname.upper(): 
						print('RETMIN - FIRST', txtVal, kk )
						return kk	
	return minElem


def _counter_returnNumericalValue( txt ):
	## filters out C1, C2 etc .since they also might end up in final numbers
	txt = txt.replace('C1','').replace('C2','').replace('C3','').replace('C4','').replace('C5','')
	_arr = txt.split()
	numArr = []
	for word in _arr:
		print('COMING into returnNumericalValue - ', word , _counter_txtisAmount( word ))
		if _counter_txtisAmount( word ) is True:
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

def _counter_returnKeyName( txt ):
	#switch_kws = [ 'SETTE', 'REJ', 'PURG', 'REMAIN', 'DISPEN', 'TOTAL' ]
	txt = txt.upper()
	if 'SETT' in txt: return 'CASSETTE'
	if 'REJ' in txt or 'PURG' in txt: return 'REJECTED'
	if 'REMAIN' in txt: return 'REMAINING'
	if 'DISPEN' in txt : return 'DISPENSED'
	if 'TOTAL' in txt : return 'TOTAL'


def _counter_returnDKey( finalReturnDict, curr_elem ):
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


def _counter_returnCtrLoc( finalReturnDict, ctr_key ):
	ll_1 = list( finalReturnDict.keys() )
	ll_1.sort()
	idx = 0
	for ele in ll_1:
		if len( finalReturnDict[ ele ] ) == 0: continue
		if ctr_key == ele: return idx
		idx += 1

def _counter_triangulate( finalReturnDict ):
	mm , adder = None, None
	## sort items by x distance now - redundant check
	numColumns = {}
	sorted_ctrs = list( finalReturnDict.keys() )
	sorted_ctrs.sort()
	totCols = 0
	typeSet = set()	
	for sckey in sorted_ctrs:	
		CD = finalReturnDict[ sckey ]
		glob_ctr = int(sckey[-1]) # Counter#0 ..so we get 0
		val_arr = list( CD.keys() )
		for ckey in val_arr:
			locArr = CD[ ckey ]
			sortD = {}
			neo_ll = []
			for elem in locArr: 
				## in case we split a string like 00000 18734 to create 2 col entries
				## they will have the same x vals and hence will result in dict entry
				## so check if key exists, bump up key of next elem by 1 ..wont matter
				## will get sorted using this val
				if elem['value_']['pts'][0] in sortD:
					innerK = int( elem['value_']['pts'][0] ) + random.randint( 2, 10 )
				else:
					innerK = int( elem['value_']['pts'][0] )
				sortD[ innerK ] = elem
			if totCols == 0:
				totCols = len( locArr )
			elif totCols != 0 and len( locArr ) != totCols:
				print('_counter_triangulate: COL NUM EXCEPTION - num cols for ',sckey\
					, ckey, ' are diff from other rows' )		
			sorted_ll = list( sortD.keys() )
			sorted_ll.sort()
			for key_ctr in range(len(sorted_ll)): 
				keys = sorted_ll[ key_ctr ]
				matrix_entry = sortD[ keys ]
				## since both Counter# and columns are now in order, start assigning
				## type # to each element ..so final struct will be key_, value_, col_hdr = type#
				matrix_entry['col_hdr_'] = 'TYPE'+str( key_ctr + 1 + glob_ctr*totCols )
				neo_ll.append( sortD[ keys ] )
				typeSet.add( 'TYPE'+str( key_ctr + 1 + glob_ctr*totCols ) )
			CD[ ckey ] = neo_ll

	## re-arrange results by TYPE	
	typeKeys = list( typeSet )
	typeKeys.sort()
	triangDResult = {}
	for t_key in typeKeys:
		for sckey in sorted_ctrs:	
			CD = finalReturnDict[ sckey ]
			ckeys = list( CD.keys() )
			for ck in ckeys:
				ele_ = CD[ ck ] ## will give row entry for CASS, REJ etc
				for arr_elem in ele_: # scroll through rows to find cols of type x
					if arr_elem['col_hdr_'] == t_key:
						if t_key in triangDResult:
							localDD = triangDResult[ t_key ]
						else:
							localDD = {}
						localDD[ ck ] = arr_elem
						triangDResult[ t_key ] = localDD	
						
	## TRIAN
	## CASSETTE + REJ = REM ; REM + DISP = TOTAL	
	for t_key in typeKeys:
		type_rec = triangDResult[ t_key ]
		typeKeys = list( type_rec.keys() )
		triangFail = False	
		for tk in typeKeys:
			try:
				type_rec[ tk ]['value'] = int( type_rec[ tk ]['value_']['text'] )
				type_rec[ tk ]['value_co_ords'] = ( type_rec[ tk ]['value_']['pts'] )
			except:
				continue		

		if len( typeKeys ) != 5: ## as shown above ..5 keys needed	
			triangFail = True
		else:
			REM = ( int( type_rec['CASSETTE']['value_']['text'] ) + \
					int( type_rec['REJECTED']['value_']['text'] ) )
			TOTAL = REM + int( type_rec['DISPENSED']['value_']['text'] )

			if int( type_rec['REMAINING']['value_']['text'] ) != REM or\
				int( type_rec['TOTAL']['value_']['text'] )  != TOTAL:
				triangFail = True
		
		if triangFail:
			type_rec['TRIANGULATION'] = 'FAIL'
		else:
			type_rec['TRIANGULATION'] = 'PASS'
	
	print( 'KERRANGG !!', triangDResult )	
	return triangDResult
		
	'''
	for key, val in finalReturnDict.items():	
		if 'CTR' not in val:
			print('FINAL DELETION - ',key, ' SINCE ITS MOSTLY EMPTY' )
			delkey.append(key)
		if None in val: del finalReturnDict[ key ][None]

	for elem in delkey: del finalReturnDict[ elem ]
	'''

def _counter_findClosest( finalReturnDict, rowLL , curr_elem , ll, minHt ):
	## rowLL - search in this curtailed list
	## elem = curr_elem
	## keyLL = assigned keys in this counter

	key_elem = _counter_retMin( rowLL, curr_elem['pts'][0], curr_elem['pts'][1], curr_elem['text'] ,minHt, ll )
	print('ENTERING _counter_findClosest ', ll )
	if len(ll) == 0 and finalReturnDict == {}:
		DKey = 'Counter#0'
		finalReturnDict[ DKey ] = {}
	elif len(ll) == 0 and finalReturnDict != {}:
		key_ll = list(finalReturnDict.keys())
		key_ll.sort()
		lastKey = int(key_ll[-1].split('#')[-1])
		#lastDKey = 'Counter#'+str(lastKey)
		DKey = 'Counter#'+str(lastKey+1)
		finalReturnDict[ DKey ] = {}
		print('BULBUL RESET LL 1', DKey )
	else:
		key_ll = list(finalReturnDict.keys())
		key_ll.sort()
		DKey = key_ll[-1]

	if ( len(ll) == 0 and key_elem is not None and \
		_counter_returnKeyName( key_elem['text'] ) == 'CASSETTE' ) or \
		( len(ll) == 0 and key_elem is None and finalReturnDict[ DKey ] == {} ):
		_typeLL = []
		key_nm   = 'CASSETTE'
		ll.append( 'CASSETTE' )
		if key_elem is None: 
			key_elem = curr_elem
		_typeLL.append( {'key_': key_elem, 'value_' : curr_elem} )
		finalReturnDict[ DKey ][ key_nm ] = _typeLL

	elif ( len(ll) > 0 ):
		cc_list = list( finalReturnDict[ DKey ].keys() )
		curr_key = False
		col_key = None
		for key_ in cc_list:
			col_list = finalReturnDict[DKey][key_] 
			row_key = None
			for loc in col_list:
				print('ITERATING loc in col_list:', loc, ' For curr ', curr_elem )
				ikey, ival = loc['key_'], loc['value_']
				if abs( curr_elem['pts'][1] - ival['pts'][1] ) <= minHt:
					curr_key = True
					row_key = loc['key_']
					col_key = key_
					print('FOUND TRUE FOR ', loc, ' For curr ', curr_elem, ' for key ', key_ )
					break
			if curr_key: break
		'''	
		if curr_key and key_elem is not None:
			kt = key_elem['text']
			if kt not in list( finalReturnDict[DKey] ):
				finalReturnDict[DKey][kt] = []
			col_list = finalReturnDict[DKey][kt]
			col_list.append( {'key_': key_elem, 'value_' : curr_elem} )
		'''	
		if curr_key:# and key_elem is None:
			col_list = finalReturnDict[DKey][col_key]
			holder = curr_elem['text'].split()
			for elem in holder:
				if _counter_txtisAmount( elem ):
					copy = curr_elem.copy()
					copy['text'] = elem
					col_list.append( {'key_': row_key, 'value_' : copy } )
		elif curr_key is False : # meaning new key
			## assume all insertions are done in order
			order = ['CASSETTE','REJECTED','REMAINING','DISPENSED','TOTAL']
			_typeLL = []
			dd_row_ll = list( finalReturnDict[DKey].keys() )
			
			key_nm   = order[ len(dd_row_ll) ]
			ll.append( key_nm )
			if key_elem is None: 
				key_elem = curr_elem
			holder = curr_elem['text'].split()
			for elem in holder:
				if _counter_txtisAmount( elem ):
					copy = curr_elem.copy()
					copy['text'] = elem
					_typeLL.append( {'key_': key_elem, 'value_' : copy} )
			finalReturnDict[ DKey ][ key_nm ] = _typeLL
		
	if 'TOTAL' in finalReturnDict[ DKey ] and \
		len( finalReturnDict[ DKey ]['TOTAL'] ) == len( finalReturnDict[ DKey ]['CASSETTE'] ):
		print('DODO SLEEP')
		ll = [] 
	
	print('HOLLERS ', finalReturnDict, ll )	
	return ll

def _counter_beginXform( _ll , fname ):
	rowD = dict()
	prevY = None
	minY = 100000
	maxY = -100000
	text_ll = []

	slipType = _counter_decideSlipType( text_ll )
	storeNumericalVals = dict()
	storeCounterVals = dict()
	minHt = 10
	masterLL = []
	for dict_ll in _ll:
		for dict_ in dict_ll:
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
			 ( 'TYPE' in locT.upper() or 'DATE' in locT.upper() or \
				'TIME' in locT.upper() or 'SETTE' in locT.upper() ):
				beginRead = True
			elif beginRead is False:
				continue 

			if _counter_txtisAmount( elem['text'] ) is False: continue
			print( 'BEGIN JOURNEY - ',locT, beginRead, elem )
			if ctr < 6: begin = 0
			else: begin = ctr - 6	
			assignedKeys = _counter_findClosest( finalReturnDict, sender[ begin: ctr ],\
									 elem, assignedKeys, minHt )
			#print( 'X Y  ', elem['text'] , elem['pts'][0], elem['pts'][1], \
			#int( elem['pts'][1] )*100+int(elem['pts'][0]), ' CLOSEST KEY = ', close_ )
			#if close_ is None: continue
			#finalReturnDict, assignedKeys = finalDecryption( assignedKeys, elem , \
			#						close_,finalReturnDict, minHt )

	##final pass 
	triangulatedRes = _counter_triangulate( finalReturnDict )

	return triangulatedRes
		
def _counter_call_api(path):
	files = {'file': open(path, 'rb')}
	op = requests.post(URL,files=files)
	kk = ast.literal_eval ( op.json() )
	_ll = kk["lines"]
	fname = (path.split('/')[-1])	
	resp = _counter_beginXform( _ll, fname )
	print( resp )

if __name__=="__main__":
    path = sys.argv[1]
    _counter_call_api(path)

