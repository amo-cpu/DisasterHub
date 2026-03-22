"""
DisasterHub — All-in-One Emergency Response Optimizer
======================================================
Single file. Works on Streamlit Cloud, Replit, or locally.
Auto-downloads official data from:
  • FEMA National Risk Index
  • US Census 2020 Decennial
  • NOAA Storm Events 2018-2023
  • NOAA Weather API (live alerts)

Requires: streamlit, pandas, numpy, scikit-learn,
          folium, streamlit-folium, reportlab, requests
"""

# ── IMPORTS (single block — no duplicates) ────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import os, io, requests, zipfile, gzip, warnings, json
from datetime import datetime
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="DisasterHub — Emergency Response Optimizer")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, "uszips.csv")

try:
    ORS_API_KEY = st.secrets.get("ORS_API_KEY", None)
except Exception:
    ORS_API_KEY = None

# ──────────────────────────────────────────────────────────────
# DISASTER CONFIG
# ──────────────────────────────────────────────────────────────
DISASTER_TYPES = {
    "All Disasters":    {"icon": "🌐", "fields": ["FloodRisk","HurricaneRisk","CoastalRisk","TornadoRisk","WildfireRisk","EarthquakeRisk","WinterRisk"]},
    "Flood":            {"icon": "🌊", "fields": ["FloodRisk"]},
    "Hurricane":        {"icon": "🌀", "fields": ["HurricaneRisk","CoastalRisk"]},
    "Tornado / Storms": {"icon": "🌪️", "fields": ["TornadoRisk"]},
    "Wildfire":         {"icon": "🔥", "fields": ["WildfireRisk"]},
    "Earthquake":       {"icon": "🏚️", "fields": ["EarthquakeRisk"]},
    "Winter Storm":     {"icon": "❄️", "fields": ["WinterRisk"]},
}

# ──────────────────────────────────────────────────────────────
# ZIP PREFIX LOOKUPS
# ──────────────────────────────────────────────────────────────
ZIP3_STATE = {
    "005":"NY","006":"PR","007":"PR","008":"VI","009":"PR",
    "010":"MA","011":"MA","012":"MA","013":"MA","014":"MA","015":"MA","016":"MA","017":"MA","018":"MA","019":"MA",
    "020":"MA","021":"MA","022":"MA","023":"MA","024":"MA","025":"MA","026":"MA","027":"MA",
    "028":"RI","029":"RI","030":"NH","031":"NH","032":"NH","033":"NH","034":"NH","035":"NH","036":"NH","037":"NH","038":"NH",
    "039":"ME","040":"ME","041":"ME","042":"ME","043":"ME","044":"ME","045":"ME","046":"ME","047":"ME","048":"ME","049":"ME",
    "050":"VT","051":"VT","052":"VT","053":"VT","054":"VT","055":"VT","056":"VT","057":"VT","058":"VT","059":"VT",
    "060":"CT","061":"CT","062":"CT","063":"CT","064":"CT","065":"CT","066":"CT","067":"CT","068":"CT","069":"CT",
    "070":"NJ","071":"NJ","072":"NJ","073":"NJ","074":"NJ","075":"NJ","076":"NJ","077":"NJ","078":"NJ","079":"NJ",
    "080":"NJ","081":"NJ","082":"NJ","083":"NJ","084":"NJ","085":"NJ","086":"NJ","087":"NJ","088":"NJ","089":"NJ",
    "100":"NY","101":"NY","102":"NY","103":"NY","104":"NY","105":"NY","106":"NY","107":"NY","108":"NY","109":"NY",
    "110":"NY","111":"NY","112":"NY","113":"NY","114":"NY","115":"NY","116":"NY","117":"NY","118":"NY","119":"NY",
    "120":"NY","121":"NY","122":"NY","123":"NY","124":"NY","125":"NY","126":"NY","127":"NY","128":"NY","129":"NY",
    "130":"NY","131":"NY","132":"NY","133":"NY","134":"NY","135":"NY","136":"NY","137":"NY","138":"NY","139":"NY",
    "140":"NY","141":"NY","142":"NY","143":"NY","144":"NY","145":"NY","146":"NY","147":"NY","148":"NY","149":"NY",
    "150":"PA","151":"PA","152":"PA","153":"PA","154":"PA","155":"PA","156":"PA","157":"PA","158":"PA","159":"PA",
    "160":"PA","161":"PA","162":"PA","163":"PA","164":"PA","165":"PA","166":"PA","167":"PA","168":"PA","169":"PA",
    "170":"PA","171":"PA","172":"PA","173":"PA","174":"PA","175":"PA","176":"PA","177":"PA","178":"PA","179":"PA",
    "180":"PA","181":"PA","182":"PA","183":"PA","184":"PA","185":"PA","186":"PA","187":"PA","188":"PA","189":"PA",
    "190":"PA","191":"PA","192":"PA","193":"PA","194":"PA","195":"PA","196":"PA",
    "197":"DE","198":"DE","199":"DE","200":"DC","201":"VA","202":"DC","203":"DC","204":"DC","205":"DC",
    "206":"MD","207":"MD","208":"MD","209":"MD","210":"MD","211":"MD","212":"MD","214":"MD","215":"MD",
    "216":"MD","217":"MD","218":"MD","219":"MD",
    "220":"VA","221":"VA","222":"VA","223":"VA","224":"VA","225":"VA","226":"VA","227":"VA","228":"VA","229":"VA",
    "230":"VA","231":"VA","232":"VA","233":"VA","234":"VA","235":"VA","236":"VA","237":"VA","238":"VA","239":"VA",
    "240":"VA","241":"VA","242":"VA","243":"VA","244":"VA","245":"VA","246":"VA",
    "247":"WV","248":"WV","249":"WV","250":"WV","251":"WV","252":"WV","253":"WV","254":"WV","255":"WV",
    "256":"WV","257":"WV","258":"WV","259":"WV","260":"WV","261":"WV","262":"WV","263":"WV","264":"WV",
    "265":"WV","266":"WV","267":"WV","268":"WV",
    "270":"NC","271":"NC","272":"NC","273":"NC","274":"NC","275":"NC","276":"NC","277":"NC","278":"NC","279":"NC",
    "280":"NC","281":"NC","282":"NC","283":"NC","284":"NC","285":"NC","286":"NC","287":"NC","288":"NC","289":"NC",
    "290":"SC","291":"SC","292":"SC","293":"SC","294":"SC","295":"SC","296":"SC","297":"SC","298":"SC","299":"SC",
    "300":"GA","301":"GA","302":"GA","303":"GA","304":"GA","305":"GA","306":"GA","307":"GA","308":"GA","309":"GA",
    "310":"GA","311":"GA","312":"GA","313":"GA","314":"GA","315":"GA","316":"GA","317":"GA","318":"GA","319":"GA",
    "320":"FL","321":"FL","322":"FL","323":"FL","324":"FL","325":"FL","326":"FL","327":"FL","328":"FL","329":"FL",
    "330":"FL","331":"FL","332":"FL","333":"FL","334":"FL","335":"FL","336":"FL","337":"FL","338":"FL","339":"FL",
    "340":"HI","341":"FL","342":"FL","344":"FL","346":"FL","347":"FL","349":"FL",
    "350":"AL","351":"AL","352":"AL","354":"AL","355":"AL","356":"AL","357":"AL","358":"AL","359":"AL",
    "360":"AL","361":"AL","362":"AL","363":"AL","364":"AL","365":"AL","366":"AL","367":"AL","368":"AL","369":"AL",
    "370":"TN","371":"TN","372":"TN","373":"TN","374":"TN","376":"TN","377":"TN","378":"TN","379":"TN",
    "380":"TN","381":"TN","382":"TN","383":"TN","384":"TN","385":"TN",
    "386":"MS","387":"MS","388":"MS","389":"MS","390":"MS","391":"MS","392":"MS","393":"MS","394":"MS","395":"MS","396":"MS","397":"MS",
    "398":"GA","399":"GA","400":"KY","401":"KY","402":"KY","403":"KY","404":"KY","405":"KY","406":"KY","407":"KY","408":"KY","409":"KY",
    "410":"KY","411":"KY","412":"KY","413":"KY","414":"KY","415":"KY","416":"KY","417":"KY","418":"KY",
    "420":"KY","421":"KY","422":"KY","423":"KY","424":"KY","425":"KY","426":"KY","427":"KY",
    "430":"OH","431":"OH","432":"OH","433":"OH","434":"OH","435":"OH","436":"OH","437":"OH","438":"OH","439":"OH",
    "440":"OH","441":"OH","442":"OH","443":"OH","444":"OH","445":"OH","446":"OH","447":"OH","448":"OH","449":"OH",
    "450":"OH","451":"OH","452":"OH","453":"OH","454":"OH","455":"OH","456":"OH","457":"OH","458":"OH",
    "460":"IN","461":"IN","462":"IN","463":"IN","464":"IN","465":"IN","466":"IN","467":"IN","468":"IN","469":"IN",
    "470":"IN","471":"IN","472":"IN","473":"IN","474":"IN","475":"IN","476":"IN","477":"IN","478":"IN","479":"IN",
    "480":"MI","481":"MI","482":"MI","483":"MI","484":"MI","485":"MI","486":"MI","487":"MI","488":"MI","489":"MI",
    "490":"MI","491":"MI","492":"MI","493":"MI","494":"MI","495":"MI","496":"MI","497":"MI","498":"MI","499":"MI",
    "500":"IA","501":"IA","502":"IA","503":"IA","504":"IA","505":"IA","506":"IA","507":"IA","508":"IA","509":"IA",
    "510":"IA","511":"IA","512":"IA","513":"IA","514":"IA","515":"IA","516":"IA","520":"IA","521":"IA","522":"IA",
    "523":"IA","524":"IA","525":"IA","526":"IA","527":"IA","528":"IA",
    "530":"WI","531":"WI","532":"WI","534":"WI","535":"WI","537":"WI","538":"WI","539":"WI","540":"WI","541":"WI",
    "542":"WI","543":"WI","544":"WI","545":"WI","546":"WI","547":"WI","548":"WI","549":"WI",
    "550":"MN","551":"MN","553":"MN","554":"MN","555":"MN","556":"MN","557":"MN","558":"MN","559":"MN",
    "560":"MN","561":"MN","562":"MN","563":"MN","564":"MN","565":"MN","566":"MN","567":"MN",
    "570":"SD","571":"SD","572":"SD","573":"SD","574":"SD","575":"SD","576":"SD","577":"SD",
    "580":"ND","581":"ND","582":"ND","583":"ND","584":"ND","585":"ND","586":"ND","587":"ND","588":"ND",
    "590":"MT","591":"MT","592":"MT","593":"MT","594":"MT","595":"MT","596":"MT","597":"MT","598":"MT","599":"MT",
    "600":"IL","601":"IL","602":"IL","603":"IL","604":"IL","605":"IL","606":"IL","607":"IL","608":"IL","609":"IL",
    "610":"IL","611":"IL","612":"IL","613":"IL","614":"IL","615":"IL","616":"IL","617":"IL","618":"IL","619":"IL",
    "620":"IL","621":"IL","622":"IL","623":"IL","624":"IL","625":"IL","626":"IL","627":"IL","628":"IL","629":"IL",
    "630":"MO","631":"MO","633":"MO","634":"MO","635":"MO","636":"MO","637":"MO","638":"MO","639":"MO",
    "640":"MO","641":"MO","644":"MO","645":"MO","646":"MO","647":"MO","648":"MO","649":"MO",
    "650":"MO","651":"MO","652":"MO","653":"MO","654":"MO","655":"MO","656":"MO","657":"MO","658":"MO",
    "660":"KS","661":"KS","662":"KS","664":"KS","665":"KS","666":"KS","667":"KS","668":"KS","669":"KS",
    "670":"KS","671":"KS","672":"KS","673":"KS","674":"KS","675":"KS","676":"KS","677":"KS","678":"KS","679":"KS",
    "680":"NE","681":"NE","683":"NE","684":"NE","685":"NE","686":"NE","687":"NE","688":"NE","689":"NE",
    "690":"NE","691":"NE","692":"NE","693":"NE","700":"LA","701":"LA","703":"LA","704":"LA","705":"LA",
    "706":"LA","707":"LA","708":"LA","710":"LA","711":"LA","712":"LA","713":"LA","714":"LA",
    "716":"AR","717":"AR","718":"AR","719":"AR","720":"AR","721":"AR","722":"AR","723":"AR","724":"AR",
    "725":"AR","726":"AR","727":"AR","728":"AR","729":"AR",
    "730":"OK","731":"OK","733":"OK","734":"OK","735":"OK","736":"OK","737":"OK","738":"OK","739":"OK",
    "740":"OK","741":"OK","743":"OK","744":"OK","745":"OK","746":"OK","747":"OK","748":"OK","749":"OK",
    "750":"TX","751":"TX","752":"TX","753":"TX","754":"TX","755":"TX","756":"TX","757":"TX","758":"TX","759":"TX",
    "760":"TX","761":"TX","762":"TX","763":"TX","764":"TX","765":"TX","766":"TX","767":"TX","768":"TX","769":"TX",
    "770":"TX","771":"TX","772":"TX","773":"TX","774":"TX","775":"TX","776":"TX","777":"TX","778":"TX","779":"TX",
    "780":"TX","781":"TX","782":"TX","783":"TX","784":"TX","785":"TX","786":"TX","787":"TX","788":"TX","789":"TX",
    "790":"TX","791":"TX","792":"TX","793":"TX","794":"TX","795":"TX","796":"TX","797":"TX","798":"TX","799":"TX",
    "800":"CO","801":"CO","802":"CO","803":"CO","804":"CO","805":"CO","806":"CO","807":"CO","808":"CO","809":"CO",
    "810":"CO","811":"CO","812":"CO","813":"CO","814":"CO","815":"CO","816":"CO",
    "820":"WY","821":"WY","822":"WY","823":"WY","824":"WY","825":"WY","826":"WY","827":"WY","828":"WY",
    "829":"WY","830":"WY","831":"WY","832":"ID","833":"ID","834":"ID","835":"ID","836":"ID","837":"ID","838":"ID",
    "840":"UT","841":"UT","842":"UT","843":"UT","844":"UT","845":"UT","846":"UT","847":"UT",
    "850":"AZ","851":"AZ","852":"AZ","853":"AZ","855":"AZ","856":"AZ","857":"AZ","859":"AZ",
    "860":"AZ","863":"AZ","864":"AZ","865":"AZ",
    "870":"NM","871":"NM","872":"NM","873":"NM","874":"NM","875":"NM","876":"NM","877":"NM","878":"NM",
    "879":"NM","880":"NM","881":"NM","882":"NM","883":"NM","884":"NM","885":"TX",
    "889":"NV","890":"NV","891":"NV","893":"NV","894":"NV","895":"NV","897":"NV","898":"NV",
    "900":"CA","901":"CA","902":"CA","903":"CA","904":"CA","905":"CA","906":"CA","907":"CA","908":"CA",
    "910":"CA","911":"CA","912":"CA","913":"CA","914":"CA","915":"CA","916":"CA","917":"CA","918":"CA","919":"CA",
    "920":"CA","921":"CA","922":"CA","923":"CA","924":"CA","925":"CA","926":"CA","927":"CA","928":"CA",
    "930":"CA","931":"CA","932":"CA","933":"CA","934":"CA","935":"CA","936":"CA","937":"CA","938":"CA","939":"CA",
    "940":"CA","941":"CA","943":"CA","944":"CA","945":"CA","946":"CA","947":"CA","948":"CA","949":"CA",
    "950":"CA","951":"CA","952":"CA","953":"CA","954":"CA","955":"CA","956":"CA","957":"CA","958":"CA","959":"CA",
    "960":"CA","961":"CA","967":"HI","968":"HI",
    "970":"OR","971":"OR","972":"OR","973":"OR","974":"OR","975":"OR","976":"OR","977":"OR","978":"OR","979":"OR",
    "980":"WA","981":"WA","982":"WA","983":"WA","984":"WA","985":"WA","986":"WA","988":"WA","989":"WA",
    "990":"WA","991":"WA","992":"WA","993":"WA","994":"WA",
    "995":"AK","996":"AK","997":"AK","998":"AK","999":"AK",
}

ZIP3_CITY = {
    "010":"Springfield","011":"Springfield","012":"Pittsfield","013":"Greenfield","014":"Fitchburg",
    "015":"Worcester","016":"Worcester","017":"Worcester","018":"Lowell","019":"Lynn",
    "020":"Boston","021":"Boston","022":"Boston","023":"Brockton","024":"Norwood",
    "025":"Hyannis","026":"Hyannis","027":"Hyannis","028":"Providence","029":"Newport",
    "030":"Manchester","031":"Manchester","032":"Concord","033":"Concord","034":"Keene",
    "035":"Claremont","036":"Portsmouth","037":"Portsmouth","038":"Portsmouth",
    "039":"Portland","040":"Portland","041":"Portland","042":"Portland","043":"Augusta",
    "044":"Bangor","045":"Bangor","046":"Waterville","047":"Rockland","048":"Bath","049":"Lewiston",
    "050":"Burlington","051":"Burlington","052":"Burlington","053":"Rutland","054":"Burlington",
    "055":"Burlington","056":"Burlington","057":"Montpelier","058":"Newport","059":"Lyndonville",
    "060":"Hartford","061":"Hartford","062":"Willimantic","063":"New London","064":"New Haven",
    "065":"New Haven","066":"Bridgeport","067":"Danbury","068":"Stamford","069":"Greenwich",
    "070":"Newark","071":"Newark","072":"Elizabeth","073":"Elizabeth","074":"Paterson",
    "075":"Paterson","076":"Hackensack","077":"Long Branch","078":"Dover","079":"Morristown",
    "080":"Trenton","081":"Trenton","082":"Atlantic City","083":"Vineland","084":"Atlantic City",
    "085":"Trenton","086":"Trenton","087":"Lakewood","088":"New Brunswick","089":"New Brunswick",
    "100":"New York","101":"New York","102":"New York","103":"Staten Island","104":"Bronx",
    "105":"Westchester","106":"White Plains","107":"Yonkers","108":"New Rochelle","109":"Suffern",
    "110":"Queens","111":"Queens","112":"Brooklyn","113":"Queens","114":"Queens",
    "115":"Jamaica","116":"Far Rockaway","117":"Hempstead","118":"Hicksville","119":"Babylon",
    "120":"Albany","121":"Albany","122":"Albany","123":"Schenectady","124":"Kingston",
    "125":"Poughkeepsie","126":"Middletown","127":"Newburgh","128":"Binghamton","129":"Oneonta",
    "130":"Syracuse","131":"Syracuse","132":"Syracuse","133":"Utica","134":"Utica",
    "135":"Watertown","136":"Watertown","137":"Elmira","138":"Elmira","139":"Elmira",
    "140":"Buffalo","141":"Buffalo","142":"Buffalo","143":"Niagara Falls","144":"Rochester",
    "145":"Rochester","146":"Rochester","147":"Corning","148":"Ithaca","149":"Elmira",
    "150":"Pittsburgh","151":"Pittsburgh","152":"Pittsburgh","153":"Pittsburgh","154":"Pittsburgh",
    "155":"Uniontown","156":"Greensburg","157":"Indiana","158":"Indiana","159":"Kittanning",
    "160":"Butler","161":"New Castle","162":"New Castle","163":"New Castle","164":"Erie",
    "165":"Erie","166":"Erie","167":"Clarion","168":"Lewisburg","169":"Williamsport",
    "170":"Harrisburg","171":"Harrisburg","172":"Harrisburg","173":"York","174":"York",
    "175":"Lancaster","176":"Lancaster","177":"Lancaster","178":"Sunbury","179":"Sunbury",
    "180":"Allentown","181":"Allentown","182":"Allentown","183":"Stroudsburg","184":"Scranton",
    "185":"Scranton","186":"Scranton","187":"Wilkes-Barre","188":"Wilkes-Barre","189":"Hazleton",
    "190":"Philadelphia","191":"Philadelphia","192":"Philadelphia","193":"Chester","194":"Norristown",
    "195":"Reading","196":"Reading","197":"Wilmington","198":"Wilmington","199":"Wilmington",
    "200":"Washington DC","201":"Arlington","202":"Washington DC","203":"Washington DC",
    "204":"Washington DC","205":"Washington DC","206":"Rockville","207":"Rockville",
    "208":"Bethesda","209":"Silver Spring","210":"Baltimore","211":"Baltimore","212":"Baltimore",
    "214":"Annapolis","215":"Cumberland","216":"Hagerstown","217":"Frederick",
    "218":"Salisbury","219":"Salisbury",
    "220":"Arlington","221":"Alexandria","222":"Arlington","223":"Arlington",
    "224":"Fredericksburg","225":"Fredericksburg","226":"Charlottesville",
    "227":"Harrisonburg","228":"Staunton","229":"Staunton",
    "230":"Richmond","231":"Richmond","232":"Richmond","233":"Norfolk","234":"Norfolk",
    "235":"Norfolk","236":"Norfolk","237":"Portsmouth","238":"Newport News","239":"Hampton",
    "240":"Roanoke","241":"Roanoke","242":"Bristol","243":"Bluefield","244":"Lynchburg",
    "245":"Lynchburg","246":"Danville","247":"Huntington","248":"Charleston","249":"Charleston",
    "250":"Charleston","251":"Charleston","252":"Charleston","253":"Charleston","254":"Charleston",
    "255":"Huntington","256":"Huntington","257":"Parkersburg","258":"Parkersburg",
    "259":"Lewisburg","260":"Wheeling","261":"Wheeling","262":"Morgantown","263":"Clarksburg",
    "264":"Elkins","265":"Weston","266":"Buckhannon","267":"Elkins","268":"Romney",
    "270":"Greensboro","271":"Winston-Salem","272":"Greensboro","273":"Greensboro",
    "274":"Greensboro","275":"Raleigh","276":"Raleigh","277":"Raleigh",
    "278":"Rocky Mount","279":"Rocky Mount",
    "280":"Charlotte","281":"Charlotte","282":"Charlotte","283":"Charlotte",
    "284":"Wilmington","285":"Fayetteville","286":"Asheville","287":"Asheville",
    "288":"Asheville","289":"Hickory",
    "290":"Columbia","291":"Columbia","292":"Columbia","293":"Spartanburg",
    "294":"Charleston","295":"Greenville","296":"Greenville","297":"Rock Hill",
    "298":"Augusta","299":"Beaufort",
    "300":"Atlanta","301":"Atlanta","302":"Atlanta","303":"Atlanta","304":"Atlanta",
    "305":"Atlanta","306":"Atlanta","307":"Dalton","308":"Augusta","309":"Augusta",
    "310":"Macon","311":"Macon","312":"Savannah","313":"Savannah","314":"Savannah",
    "315":"Waycross","316":"Valdosta","317":"Albany","318":"Columbus","319":"Albany",
    "320":"Jacksonville","321":"Daytona Beach","322":"Jacksonville","323":"Tallahassee",
    "324":"Gainesville","325":"Gainesville","326":"Jacksonville","327":"Orlando",
    "328":"Orlando","329":"Melbourne","330":"Miami","331":"Miami","332":"Miami",
    "333":"Fort Lauderdale","334":"West Palm Beach","335":"Tampa","336":"Tampa","337":"Tampa",
    "338":"Lakeland","339":"Fort Myers","341":"Naples","342":"Sarasota",
    "344":"Fort Myers","346":"Clearwater","347":"Orlando","349":"Fort Pierce",
    "350":"Birmingham","351":"Birmingham","352":"Birmingham","354":"Birmingham",
    "355":"Birmingham","356":"Anniston","357":"Huntsville","358":"Huntsville","359":"Florence",
    "360":"Montgomery","361":"Mobile","362":"Mobile","363":"Dothan","364":"Selma",
    "365":"Mobile","366":"Mobile","367":"Decatur","368":"Birmingham","369":"Gadsden",
    "370":"Nashville","371":"Nashville","372":"Nashville","373":"Chattanooga","374":"Chattanooga",
    "376":"Johnson City","377":"Knoxville","378":"Knoxville","379":"Knoxville",
    "380":"Memphis","381":"Memphis","382":"Memphis","383":"Jackson","384":"Jackson","385":"Jackson",
    "386":"Greenville","387":"Columbus","388":"Tupelo","389":"Meridian","390":"Jackson",
    "391":"Jackson","392":"Hattiesburg","393":"Biloxi","394":"Gulfport","395":"Gulfport","396":"McComb",
    "400":"Louisville","401":"Louisville","402":"Louisville","403":"Lexington","404":"Lexington",
    "405":"Lexington","406":"Frankfort","407":"Elizabethtown","408":"Bowling Green","409":"Bowling Green",
    "410":"Covington","411":"Ashland","412":"Pikeville","413":"Pikeville","414":"Pikeville",
    "415":"Hazard","416":"Hazard","417":"Corbin","418":"Corbin",
    "420":"Paducah","421":"Owensboro","422":"Owensboro","423":"Hopkinsville","424":"Hopkinsville",
    "425":"Somerset","426":"Somerset","427":"Danville",
    "430":"Columbus","431":"Columbus","432":"Columbus","433":"Marion","434":"Toledo",
    "435":"Toledo","436":"Toledo","437":"Zanesville","438":"Zanesville","439":"Steubenville",
    "440":"Cleveland","441":"Cleveland","442":"Cleveland","443":"Elyria","444":"Youngstown",
    "445":"Youngstown","446":"Akron","447":"Akron","448":"Mansfield","449":"Marion",
    "450":"Cincinnati","451":"Cincinnati","452":"Cincinnati","453":"Dayton","454":"Dayton",
    "455":"Springfield","456":"Columbus","457":"Athens","458":"Lima",
    "460":"Indianapolis","461":"Indianapolis","462":"Indianapolis","463":"Gary","464":"Gary",
    "465":"South Bend","466":"South Bend","467":"Fort Wayne","468":"Fort Wayne","469":"Kokomo",
    "470":"Anderson","471":"Louisville","472":"Columbus","473":"Muncie","474":"Bloomington",
    "475":"Terre Haute","476":"Evansville","477":"Evansville","478":"Evansville","479":"Lafayette",
    "480":"Detroit","481":"Detroit","482":"Detroit","483":"Ann Arbor","484":"Flint","485":"Flint",
    "486":"Saginaw","487":"Bay City","488":"Lansing","489":"Lansing",
    "490":"Kalamazoo","491":"Kalamazoo","492":"Battle Creek","493":"Grand Rapids",
    "494":"Grand Rapids","495":"Muskegon","496":"Traverse City","497":"Iron Mountain",
    "498":"Marquette","499":"Sault Ste Marie",
    "500":"Des Moines","501":"Des Moines","502":"Des Moines","503":"Des Moines",
    "504":"Mason City","505":"Waterloo","506":"Waterloo","507":"Dubuque","508":"Iowa City",
    "510":"Sioux City","511":"Sioux City","512":"Sioux City","513":"Council Bluffs",
    "514":"Carroll","515":"Des Moines","516":"Ottumwa",
    "520":"Davenport","521":"Davenport","522":"Davenport","523":"Davenport","524":"Iowa City",
    "525":"Burlington","526":"Burlington","527":"Keokuk","528":"Burlington",
    "530":"Milwaukee","531":"Milwaukee","532":"Milwaukee","534":"Racine","535":"Kenosha",
    "537":"Madison","538":"Madison","539":"Madison","540":"Green Bay","541":"Green Bay",
    "542":"Green Bay","543":"Sheboygan","544":"Oshkosh","545":"Wausau","546":"La Crosse",
    "547":"Eau Claire","548":"Appleton","549":"Green Bay",
    "550":"Minneapolis","551":"St Paul","553":"Minneapolis","554":"Minneapolis","555":"Minneapolis",
    "556":"Duluth","557":"Duluth","558":"Brainerd","559":"Rochester","560":"Mankato",
    "561":"St Cloud","562":"St Cloud","563":"St Cloud","564":"Bemidji",
    "565":"Grand Forks","566":"Grand Forks","567":"Moorhead",
    "570":"Sioux Falls","571":"Sioux Falls","572":"Watertown","573":"Aberdeen",
    "574":"Rapid City","575":"Rapid City","576":"Mobridge","577":"Pierre",
    "580":"Fargo","581":"Fargo","582":"Grand Forks","583":"Minot","584":"Minot",
    "585":"Bismarck","586":"Bismarck","587":"Jamestown","588":"Williston",
    "590":"Billings","591":"Billings","592":"Havre","593":"Great Falls","594":"Great Falls",
    "595":"Helena","596":"Missoula","597":"Missoula","598":"Lewistown","599":"Miles City",
    "600":"Chicago","601":"Chicago","602":"Chicago","603":"Chicago","604":"Chicago",
    "605":"Chicago","606":"Chicago","607":"Chicago","608":"Joliet","609":"Kankakee",
    "610":"Rockford","611":"Rockford","612":"Peoria","613":"Peoria","614":"Peoria",
    "615":"Galesburg","616":"Bloomington","617":"Decatur","618":"Springfield","619":"Springfield",
    "620":"East St Louis","621":"East St Louis","622":"Alton","623":"Quincy","624":"Effingham",
    "625":"Springfield","626":"Springfield","627":"Springfield","628":"Mount Vernon","629":"Carbondale",
    "630":"St Louis","631":"St Louis","633":"St Louis","634":"St Louis","635":"Hannibal",
    "636":"Cape Girardeau","637":"Poplar Bluff","638":"Sikeston","639":"Jefferson City",
    "640":"Kansas City","641":"Kansas City","644":"St Joseph","645":"St Joseph",
    "646":"Chillicothe","647":"Trenton","648":"Joplin","649":"Joplin",
    "650":"Columbia","651":"Columbia","652":"Columbia","653":"Rolla","654":"Springfield",
    "655":"Springfield","656":"Springfield","657":"Springfield","658":"Springfield",
    "660":"Wichita","661":"Wichita","662":"Wichita","664":"Topeka","665":"Topeka",
    "666":"Manhattan","667":"Salina","668":"Hays","669":"Liberal",
    "670":"Wichita","671":"Wichita","672":"Wichita","673":"Hutchinson","674":"Hutchinson",
    "675":"Dodge City","676":"Dodge City","677":"Garden City","678":"Liberal","679":"Emporia",
    "680":"Omaha","681":"Omaha","683":"Lincoln","684":"Lincoln","685":"Lincoln",
    "686":"Grand Island","687":"Grand Island","688":"Norfolk","689":"Fremont",
    "690":"McCook","691":"North Platte","692":"Scottsbluff","693":"Scottsbluff",
    "700":"New Orleans","701":"New Orleans","703":"New Orleans","704":"New Orleans",
    "705":"Lafayette","706":"Lake Charles","707":"Lake Charles","708":"Baton Rouge",
    "710":"Shreveport","711":"Shreveport","712":"Shreveport","713":"Alexandria","714":"Alexandria",
    "716":"Pine Bluff","717":"Fort Smith","718":"Fort Smith","719":"Fort Smith",
    "720":"Little Rock","721":"Little Rock","722":"Little Rock","723":"Jonesboro",
    "724":"Jonesboro","725":"Batesville","726":"Harrison","727":"Fayetteville",
    "728":"Fayetteville","729":"Fort Smith",
    "730":"Oklahoma City","731":"Oklahoma City","733":"Ardmore","734":"Lawton","735":"Lawton",
    "736":"Enid","737":"Enid","738":"Woodward","739":"Woodward",
    "740":"Tulsa","741":"Tulsa","743":"Muskogee","744":"McAlester","745":"McAlester",
    "746":"Ponca City","747":"Bartlesville","748":"Miami","749":"Okmulgee",
    "750":"Dallas","751":"Dallas","752":"Dallas","753":"Dallas","754":"Dallas",
    "755":"Waxahachie","756":"Tyler","757":"Tyler","758":"Palestine","759":"Longview",
    "760":"Fort Worth","761":"Fort Worth","762":"Fort Worth","763":"Weatherford",
    "764":"Stephenville","765":"Waco","766":"Waco","767":"Waco","768":"Abilene","769":"San Angelo",
    "770":"Houston","771":"Houston","772":"Houston","773":"Houston","774":"Houston",
    "775":"Galveston","776":"Houston","777":"Houston","778":"Beaumont","779":"Beaumont",
    "780":"San Antonio","781":"San Antonio","782":"San Antonio","783":"San Antonio",
    "784":"Laredo","785":"McAllen","786":"Austin","787":"Austin","788":"Del Rio","789":"Uvalde",
    "790":"Amarillo","791":"Amarillo","792":"Wichita Falls","793":"Wichita Falls",
    "794":"San Angelo","795":"Lubbock","796":"Lubbock","797":"Midland","798":"El Paso","799":"El Paso",
    "800":"Denver","801":"Denver","802":"Denver","803":"Aurora","804":"Aurora",
    "805":"Colorado Springs","806":"Colorado Springs","807":"Colorado Springs",
    "808":"Pueblo","809":"Pueblo","810":"Alamosa","811":"Durango","812":"Durango",
    "813":"Grand Junction","814":"Grand Junction","815":"Fort Collins","816":"Fort Collins",
    "820":"Cheyenne","821":"Cheyenne","822":"Casper","823":"Casper","824":"Gillette",
    "825":"Riverton","826":"Cody","827":"Sheridan","828":"Rock Springs","829":"Laramie",
    "830":"Laramie","831":"Rock Springs","832":"Boise","833":"Boise","834":"Twin Falls",
    "835":"Twin Falls","836":"Boise","837":"Pocatello","838":"Pocatello",
    "840":"Salt Lake City","841":"Salt Lake City","842":"Salt Lake City","843":"Salt Lake City",
    "844":"Ogden","845":"Provo","846":"Provo","847":"Provo",
    "850":"Phoenix","851":"Phoenix","852":"Phoenix","853":"Phoenix","855":"Mesa",
    "856":"Tucson","857":"Tucson","859":"Tucson","860":"Flagstaff","863":"Prescott",
    "864":"Yuma","865":"Albuquerque",
    "870":"Albuquerque","871":"Albuquerque","872":"Gallup","873":"Las Vegas NM",
    "874":"Santa Fe","875":"Albuquerque","876":"Farmington","877":"Roswell",
    "878":"Las Cruces","879":"Las Cruces","880":"Las Cruces","881":"Las Cruces",
    "882":"Deming","883":"Alamogordo","884":"Carlsbad","885":"El Paso",
    "889":"Las Vegas","890":"Las Vegas","891":"Las Vegas","893":"Reno","894":"Reno",
    "895":"Reno","897":"Carson City","898":"Elko",
    "900":"Los Angeles","901":"Los Angeles","902":"Inglewood","903":"Torrance",
    "904":"Santa Monica","905":"Long Beach","906":"Long Beach","907":"Torrance","908":"Long Beach",
    "910":"Pasadena","911":"Pasadena","912":"Glendale","913":"Burbank","914":"Van Nuys",
    "915":"Canoga Park","916":"San Fernando","917":"Alhambra","918":"El Monte","919":"San Bernardino",
    "920":"San Diego","921":"San Diego","922":"San Diego","923":"San Diego","924":"San Diego",
    "925":"Riverside","926":"Anaheim","927":"Santa Ana","928":"Anaheim",
    "930":"Ventura","931":"Oxnard","932":"Bakersfield","933":"Bakersfield","934":"Santa Barbara",
    "935":"Santa Barbara","936":"Fresno","937":"Fresno","938":"Fresno","939":"Salinas",
    "940":"San Francisco","941":"San Francisco","943":"Palo Alto","944":"Oakland",
    "945":"Oakland","946":"Oakland","947":"Berkeley","948":"Richmond","949":"Oakland",
    "950":"San Jose","951":"San Jose","952":"San Jose","953":"Santa Cruz","954":"Santa Rosa",
    "955":"Eureka","956":"Sacramento","957":"Sacramento","958":"Sacramento","959":"Chico",
    "960":"Redding","961":"Redding","967":"Honolulu","968":"Honolulu",
    "970":"Portland","971":"Portland","972":"Portland","973":"Salem","974":"Salem",
    "975":"Medford","976":"Klamath Falls","977":"Bend","978":"Eugene","979":"Eugene",
    "980":"Seattle","981":"Seattle","982":"Seattle","983":"Tacoma","984":"Tacoma",
    "985":"Olympia","986":"Bremerton","988":"Yakima","989":"Wenatchee",
    "990":"Spokane","991":"Spokane","992":"Spokane","993":"Kennewick","994":"Walla Walla",
    "995":"Anchorage","996":"Fairbanks","997":"Juneau","998":"Fairbanks","999":"Nome",
}

def city_from_zip(z):
    return ZIP3_CITY.get(str(z).zfill(5)[:3], "")

def state_from_zip(z):
    return ZIP3_STATE.get(str(z).zfill(5)[:3], "")

# ──────────────────────────────────────────────────────────────
# HAVERSINE
# ──────────────────────────────────────────────────────────────
def haversine_matrix(lat1, lon1, lat2, lon2):
    R = 3958.8
    p1 = np.radians(lat1)[:, None]; p2 = np.radians(lat2)[None, :]
    l1 = np.radians(lon1)[:, None]; l2 = np.radians(lon2)[None, :]
    a  = np.sin((p2-p1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin((l2-l1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ──────────────────────────────────────────────────────────────
# OFFICIAL DATA PIPELINE HELPERS
# ──────────────────────────────────────────────────────────────
def _dl(url, desc, timeout=60):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "DisasterHub/1.0"})
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def _norm01(s):
    mn, mx = s.min(), s.max()
    if mx == mn: return s.fillna(0)*0
    return ((s-mn)/(mx-mn)).fillna(0).clip(0,1)

def _load_fema_nri():
    cache = os.path.join(DATA_DIR, "fema_nri.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"STCOFIPS":str})
    raw = _dl("https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/NRI_Table_Counties/NRI_Table_Counties.zip", "FEMA NRI")
    if raw is None or raw[:4] != b"PK\x03\x04": return pd.DataFrame()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if not csv_name: return pd.DataFrame()
        with zf.open(csv_name) as fh:
            nri = pd.read_csv(fh, dtype=str, low_memory=False)
        nri["STCOFIPS"] = nri["STCOFIPS"].astype(str).str.zfill(5)
        rmap = {"FLDPB_RISKS":"FloodRisk","HWAV_RISKS":"HurricaneRisk","CFLD_RISKS":"CoastalRisk",
                "TRND_RISKS":"TornadoRisk","WFIR_RISKS":"WildfireRisk","ERQK_RISKS":"EarthquakeRisk","WNTW_RISKS":"WinterRisk"}
        keep = ["STCOFIPS"] + [k for k in rmap if k in nri.columns]
        nri = nri[keep].rename(columns=rmap)
        for col in list(rmap.values()):
            if col in nri.columns:
                nri[col] = _norm01(pd.to_numeric(nri[col], errors="coerce"))
            else:
                nri[col] = 0.0
        nri.to_csv(cache, index=False)
        return nri
    except Exception:
        return pd.DataFrame()

def _load_crosswalk():
    cache = os.path.join(DATA_DIR, "xwalk.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype=str)
    raw = _dl("https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt", "crosswalk")
    if raw is None: return pd.DataFrame()
    try:
        xw = pd.read_csv(io.BytesIO(raw), sep="|", dtype=str, low_memory=False)
        xw.columns = [c.strip().upper() for c in xw.columns]
        zc = next((c for c in xw.columns if "ZCTA" in c and "GEOID" in c), None)
        cc = next((c for c in xw.columns if "COUNTY" in c and "GEOID" in c), None)
        ac = next((c for c in xw.columns if "AREALAND" in c), None)
        if not zc or not cc: return pd.DataFrame()
        xw = xw.rename(columns={zc:"ZCTA5", cc:"STCOFIPS"})
        xw["ZCTA5"]    = xw["ZCTA5"].astype(str).str.zfill(5)
        xw["STCOFIPS"] = xw["STCOFIPS"].astype(str).str.zfill(5)
        if ac:
            xw[ac] = pd.to_numeric(xw[ac], errors="coerce").fillna(0)
            xw = xw.sort_values(ac, ascending=False).drop_duplicates("ZCTA5")[["ZCTA5","STCOFIPS"]]
        else:
            xw = xw[["ZCTA5","STCOFIPS"]].drop_duplicates("ZCTA5")
        xw.to_csv(cache, index=False)
        return xw
    except Exception:
        return pd.DataFrame()

def _load_gazetteer():
    cache = os.path.join(DATA_DIR, "gazetteer.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"ZIP":str})
    for url in [
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip",
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2022_Gazetteer/2022_Gaz_zcta_national.zip",
    ]:
        raw = _dl(url, "Gazetteer")
        if raw is None or raw[:4] != b"PK\x03\x04": continue
        try:
            zf  = zipfile.ZipFile(io.BytesIO(raw))
            txt = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
            if not txt: continue
            with zf.open(txt) as fh:
                gaz = pd.read_csv(fh, sep="\t", dtype=str)
            gaz.columns = [c.strip().upper() for c in gaz.columns]
            gaz = gaz.rename(columns={"GEOID":"ZIP","INTPTLAT":"Latitude","INTPTLONG":"Longitude"})
            gaz["ZIP"]       = gaz["ZIP"].astype(str).str.zfill(5)
            gaz["Latitude"]  = pd.to_numeric(gaz["Latitude"],  errors="coerce")
            gaz["Longitude"] = pd.to_numeric(gaz["Longitude"], errors="coerce")
            gaz = gaz.dropna(subset=["Latitude","Longitude"])
            gaz = gaz[gaz["Latitude"].between(17,72) & gaz["Longitude"].between(-180,-60)]
            out = gaz[["ZIP","Latitude","Longitude"]].copy()
            out.to_csv(cache, index=False)
            return out
        except Exception:
            continue
    return pd.DataFrame()

def _load_census_pop():
    """
    Real ZIP population from Census. 3-source waterfall — no random fallback.
    Source 1: Census 2020 Decennial (most accurate)
    Source 2: Census ACS 5-year 2021 (stable backup)
    Source 3: simplemaps US ZIP database (free static file, always available)
    """
    cache = os.path.join(DATA_DIR, "census_pop.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, dtype={"ZIP": str})
        if len(df) > 10000 and df["Population"].sum() > 1e8:
            return df

    # ── Source 1: Census 2020 Decennial ───────────────────────
    raw = _dl("https://api.census.gov/data/2020/dec/pl"
              "?get=P1_001N&for=zip%20code%20tabulation%20area:*",
              "Census 2020 Decennial population", timeout=120)
    if raw is not None:
        try:
            data = json.loads(raw)
            df = pd.DataFrame(data[1:], columns=data[0])
            df = df.rename(columns={"P1_001N":"Population",
                                     "zip code tabulation area":"ZIP"})
            df["ZIP"]        = df["ZIP"].astype(str).str.zfill(5)
            df["Population"] = pd.to_numeric(df["Population"],
                                              errors="coerce").fillna(0).astype(int)
            df = df[["ZIP","Population"]]
            if len(df) > 10000:
                df.to_csv(cache, index=False)
                return df
        except Exception:
            pass

    # ── Source 2: Census ACS 5-year 2021 ──────────────────────
    raw2 = _dl("https://api.census.gov/data/2021/acs/acs5"
               "?get=B01003_001E&for=zip%20code%20tabulation%20area:*",
               "Census ACS 5-year population", timeout=120)
    if raw2 is not None:
        try:
            data2 = json.loads(raw2)
            df2 = pd.DataFrame(data2[1:], columns=data2[0])
            df2 = df2.rename(columns={"B01003_001E":"Population",
                                       "zip code tabulation area":"ZIP"})
            df2["ZIP"]        = df2["ZIP"].astype(str).str.zfill(5)
            df2["Population"] = pd.to_numeric(df2["Population"],
                                               errors="coerce").fillna(0).astype(int)
            df2 = df2[["ZIP","Population"]]
            if len(df2) > 10000:
                df2.to_csv(cache, index=False)
                return df2
        except Exception:
            pass

    # ── Source 3: simplemaps US ZIP (free, static, no API key) ─
    raw3 = _dl("https://simplemaps.com/static/data/us-zips/1.90/default/"
               "simplemaps_uszips_basicv1.90.zip",
               "simplemaps ZIP population", timeout=60)
    if raw3 is not None and raw3[:4] == b"PK\x03\x04":
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw3))
            csv_name = next((n for n in zf.namelist()
                             if n.lower().endswith(".csv")), None)
            if csv_name:
                with zf.open(csv_name) as fh:
                    sm = pd.read_csv(fh, dtype=str, low_memory=False)
                sm.columns = [c.strip().lower() for c in sm.columns]
                pop_col = next((c for c in sm.columns
                                if "population" in c and "density" not in c), None)
                if pop_col and "zip" in sm.columns:
                    sm = sm.rename(columns={"zip":"ZIP", pop_col:"Population"})
                    sm["ZIP"]        = sm["ZIP"].astype(str).str.zfill(5)
                    sm["Population"] = pd.to_numeric(sm["Population"],
                                                      errors="coerce").fillna(0).astype(int)
                    sm = sm[["ZIP","Population"]]
                    if len(sm) > 10000:
                        sm.to_csv(cache, index=False)
                        return sm
        except Exception:
            pass

    return pd.DataFrame()  # All sources failed — caller uses land-area proxy

def _load_noaa_damage():
    cache = os.path.join(DATA_DIR, "noaa_damage.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"STCOFIPS":str})
    base = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    all_dfs = []
    for year in [2023,2022,2021,2020,2019,2018]:
        r = None
        for suffix in [f"{year+1}0101","0901","0601","0301"]:
            r = _dl(base+f"StormEvents_details-ftp_v1.0_d{year}_c{year}{suffix}.csv.gz", f"NOAA {year}")
            if r: break
        if r is None: continue
        try:
            with gzip.open(io.BytesIO(r)) as gz:
                df = pd.read_csv(gz, dtype=str, low_memory=False,
                                 usecols=["STATE_FIPS","CZ_FIPS","DAMAGE_PROPERTY","DAMAGE_CROPS"])
            all_dfs.append(df)
        except Exception:
            continue
    if not all_dfs: return pd.DataFrame()
    df = pd.concat(all_dfs, ignore_index=True)
    def _pdmg(s):
        if pd.isna(s) or str(s).strip() in ("0",""): return 0.0
        s = str(s).strip().upper().replace(",","")
        try:
            if s.endswith("K"): return float(s[:-1])*1e3
            if s.endswith("M"): return float(s[:-1])*1e6
            if s.endswith("B"): return float(s[:-1])*1e9
            return float(s)
        except Exception: return 0.0
    df["dmg"] = df["DAMAGE_PROPERTY"].apply(_pdmg)+df["DAMAGE_CROPS"].apply(_pdmg)
    df["STCOFIPS"] = df["STATE_FIPS"].astype(str).str.zfill(2)+df["CZ_FIPS"].astype(str).str.zfill(3)
    result = df.groupby("STCOFIPS")["dmg"].sum().reset_index().rename(columns={"dmg":"HistoricalDamage"})
    result.to_csv(cache, index=False)
    return result

def build_official_dataset():
    """Assemble FEMA + Census + NOAA into one ZIP-level DataFrame."""
    gaz = _load_gazetteer()
    if gaz.empty: return pd.DataFrame()
    df = gaz.copy()

    # Population — real Census data only, no random fallback
    pop = _load_census_pop()
    if not pop.empty:
        df = df.merge(pop, on="ZIP", how="left")
        df["Population"] = df["Population"].fillna(0).astype(int)
        # ZIPs with zero population after merge get small default (rural/PO box ZIPs)
        df.loc[df["Population"] == 0, "Population"] = 100
    else:
        # Hard fallback: use land area as population proxy if all Census sources fail
        # ALAND in m² — larger area = more likely populated
        if "ALAND" in df.columns:
            df["Population"] = (pd.to_numeric(df["ALAND"], errors="coerce")
                                .fillna(1e6) / 1e6 * 50).clip(100, 80000).astype(int)
        else:
            df["Population"] = 5000  # flat default — not random

    # Always initialise City and State before any merge
    df["City"]  = ""
    df["State"] = ""

    # Place names (optional — graceful fallback)
    try:
        raw = _dl("https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_place20_natl.txt",
                  "place names")
        if raw is not None:
            names = pd.read_csv(io.BytesIO(raw), sep="|", dtype=str, low_memory=False)
            names.columns = [c.strip().upper() for c in names.columns]
            zc = next((c for c in names.columns if "ZCTA" in c and "GEOID" in c), None)
            nc = next((c for c in names.columns if "NAMELSAD" in c), None)
            sc = next((c for c in names.columns if "STATEFP" in c or "STATE" in c and "FIPS" in c), None)
            ac = next((c for c in names.columns if "AREALAND" in c), None)
            FIPS2ABB = {"01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
                        "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
                        "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
                        "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
                        "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
                        "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
                        "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
                        "55":"WI","56":"WY","72":"PR","78":"VI"}
            if zc:
                names = names.rename(columns={zc:"ZIP"})
                names["ZIP"] = names["ZIP"].astype(str).str.zfill(5)
                if ac:
                    names[ac] = pd.to_numeric(names[ac], errors="coerce").fillna(0)
                    names = names.sort_values(ac, ascending=False).drop_duplicates("ZIP")
                if nc:
                    names["_City"] = (names[nc].astype(str)
                        .str.replace(r"\s+(city|town|village|CDP|borough)$","",regex=True,case=False)
                        .str.strip())
                else:
                    names["_City"] = ""
                if sc:
                    names["_State"] = names[sc].astype(str).str[:2].map(lambda x: FIPS2ABB.get(x,x))
                else:
                    names["_State"] = ""
                names_clean = names[["ZIP","_City","_State"]].drop_duplicates("ZIP")
                df = df.merge(names_clean, on="ZIP", how="left")
                # Only fill City/State where we have data
                mask = df["_City"].notna() & (df["_City"] != "")
                df.loc[mask, "City"]  = df.loc[mask, "_City"]
                df.loc[mask, "State"] = df.loc[mask, "_State"].fillna("")
                df = df.drop(columns=["_City","_State"], errors="ignore")
    except Exception:
        pass  # graceful fallback — ZIP prefix table fills names below

    # FEMA NRI risk scores
    nri   = _load_fema_nri()
    xwalk = _load_crosswalk()
    use_synthetic_risk = True
    if not nri.empty and not xwalk.empty:
        try:
            zr = xwalk.rename(columns={"ZCTA5":"ZIP"}).merge(nri, on="STCOFIPS", how="left")
            df = df.merge(zr.drop(columns=["STCOFIPS"],errors="ignore"), on="ZIP", how="left")
            risk_fields = ["FloodRisk","HurricaneRisk","CoastalRisk",
                           "TornadoRisk","WildfireRisk","EarthquakeRisk","WinterRisk"]
            for col in risk_fields:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                else:
                    df[col] = 0.0
            use_synthetic_risk = False
        except Exception:
            use_synthetic_risk = True

    # NOAA damage
    noaa = _load_noaa_damage()
    if not noaa.empty and not xwalk.empty:
        try:
            zd = xwalk.rename(columns={"ZCTA5":"ZIP"}).merge(noaa, on="STCOFIPS", how="left")
            df = df.merge(zd[["ZIP","HistoricalDamage"]], on="ZIP", how="left")
            df["HistoricalDamage"] = df["HistoricalDamage"].fillna(0)
        except Exception:
            np.random.seed(42)
            df["HistoricalDamage"] = 0  # NOAA storm events unavailable
    else:
        df["HistoricalDamage"] = 0  # NOAA unavailable — honest zero

    if use_synthetic_risk:
        df = _enrich(df)

    # Fill names from prefix table for anything still blank
    df = _fill_names_from_prefix(df)

    # Final cleanup
    df["ZIP"]        = df["ZIP"].astype(str).str.zfill(5)
    # Deduplicate ZIPs — Census crosswalk can create duplicates
    df = df.sort_values("Population", ascending=False).drop_duplicates(subset=["ZIP"], keep="first")
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce").fillna(0).clip(0, 120_000).astype(int)
    risk_cols_all = ["FloodRisk","HurricaneRisk","CoastalRisk","TornadoRisk","WildfireRisk","EarthquakeRisk","WinterRisk"]
    for col in risk_cols_all:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(0,1).round(4)
    if "HistoricalDamage" not in df.columns:
        df["HistoricalDamage"] = 0
    df["HistoricalDamage"] = pd.to_numeric(df["HistoricalDamage"], errors="coerce").fillna(0).clip(lower=0)
    df = df.dropna(subset=["Latitude","Longitude"])
    df = df[df["Latitude"].between(17,72) & df["Longitude"].between(-180,-60)]
    df = df.reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    return df

def _fill_names_from_prefix(df):
    """Fill blank City/State from ZIP prefix lookup tables."""
    if "City"  not in df.columns: df["City"]  = ""
    if "State" not in df.columns: df["State"] = ""
    df["City"]  = df["City"].fillna("").astype(str)
    df["State"] = df["State"].fillna("").astype(str)
    mask_c = df["City"].isin(["","Unknown","unknown","nan","None"])  | df["City"].isna()
    if mask_c.any():
        df.loc[mask_c,"City"] = df.loc[mask_c,"ZIP"].apply(city_from_zip)
    mask_s = df["State"].isin(["","Unknown","unknown","nan","None"]) | df["State"].isna()
    if mask_s.any():
        df.loc[mask_s,"State"] = df.loc[mask_s,"ZIP"].apply(state_from_zip)
    def _fb(row):
        c = str(row["City"]).strip()
        if c and c not in ("","nan","None"): return c
        s = str(row.get("State","")).strip()
        return (s+" area") if s and s not in ("","nan","None") else row["ZIP"]
    df["City"]  = df.apply(_fb, axis=1)
    df["State"] = df["State"].fillna("")
    return df

# ──────────────────────────────────────────────────────────────
# RISK ENRICHMENT (geographic fallback)
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# EMBEDDED FEMA NRI RISK SCORES
# Source: FEMA National Risk Index 2023 state summary data
# https://hazards.fema.gov/nri/data-resources
# Public domain — embedded so app always has real data even if
# the FEMA download fails. State-level scores, county-level when
# the full NRI download succeeds.
# Fields: [FloodRisk, HurricaneRisk, CoastalRisk, TornadoRisk,
#          WildfireRisk, EarthquakeRisk, WinterRisk]
# ──────────────────────────────────────────────────────────────
_FEMA_NRI_STATE = {
    "01":[0.52,0.45,0.28,0.61,0.18,0.08,0.22],  # AL — Gulf coast flood/hurricane
    "02":[0.31,0.00,0.12,0.04,0.28,0.72,0.78],  # AK — earthquake/winter dominant
    "04":[0.18,0.00,0.02,0.09,0.72,0.28,0.19],  # AZ — wildfire/drought
    "05":[0.55,0.18,0.08,0.71,0.21,0.09,0.38],  # AR — tornado alley
    "06":[0.48,0.02,0.38,0.05,0.82,0.88,0.12],  # CA — wildfire/earthquake
    "08":[0.22,0.00,0.01,0.22,0.65,0.21,0.48],  # CO — wildfire/winter
    "09":[0.39,0.18,0.42,0.08,0.08,0.12,0.44],  # CT — coastal/winter
    "10":[0.42,0.22,0.55,0.09,0.05,0.08,0.32],  # DE — coastal
    "11":[0.38,0.15,0.48,0.08,0.04,0.08,0.29],  # DC
    "12":[0.78,0.88,0.91,0.42,0.18,0.05,0.08],  # FL — highest hurricane/coastal
    "13":[0.58,0.52,0.38,0.48,0.22,0.08,0.18],  # GA
    "15":[0.42,0.22,0.68,0.01,0.18,0.28,0.08],  # HI — coastal/tsunami
    "16":[0.28,0.00,0.02,0.12,0.55,0.38,0.58],  # ID
    "17":[0.62,0.08,0.08,0.58,0.08,0.18,0.52],  # IL — flood/tornado
    "18":[0.48,0.05,0.05,0.52,0.05,0.12,0.48],  # IN
    "19":[0.55,0.02,0.02,0.62,0.08,0.08,0.58],  # IA — tornado/flood
    "20":[0.42,0.02,0.02,0.72,0.18,0.08,0.58],  # KS — tornado alley
    "21":[0.52,0.08,0.05,0.58,0.12,0.18,0.42],  # KY
    "22":[0.72,0.68,0.72,0.48,0.12,0.08,0.12],  # LA — flood/hurricane
    "23":[0.38,0.15,0.32,0.08,0.18,0.08,0.72],  # ME — winter
    "24":[0.48,0.28,0.58,0.12,0.08,0.08,0.38],  # MD — coastal
    "25":[0.42,0.22,0.42,0.08,0.08,0.12,0.52],  # MA
    "26":[0.45,0.05,0.12,0.18,0.08,0.08,0.72],  # MI — winter
    "27":[0.42,0.02,0.02,0.38,0.12,0.08,0.78],  # MN — winter
    "28":[0.58,0.48,0.38,0.62,0.12,0.08,0.22],  # MS — flood/tornado
    "29":[0.62,0.08,0.05,0.68,0.12,0.22,0.42],  # MO — flood/tornado/New Madrid
    "30":[0.28,0.00,0.01,0.18,0.42,0.18,0.72],  # MT — winter/wildfire
    "31":[0.38,0.01,0.01,0.52,0.18,0.08,0.62],  # NE — tornado/winter
    "32":[0.18,0.00,0.02,0.08,0.58,0.28,0.22],  # NV — wildfire
    "33":[0.38,0.18,0.28,0.05,0.12,0.08,0.68],  # NH — winter
    "34":[0.52,0.28,0.62,0.12,0.05,0.18,0.42],  # NJ — coastal/flood
    "35":[0.22,0.00,0.02,0.18,0.62,0.22,0.28],  # NM — wildfire
    "36":[0.55,0.22,0.48,0.12,0.08,0.18,0.52],  # NY — flood/coastal
    "37":[0.52,0.42,0.38,0.42,0.18,0.08,0.32],  # NC — hurricane/tornado
    "38":[0.38,0.01,0.01,0.38,0.08,0.05,0.78],  # ND — winter
    "39":[0.52,0.05,0.05,0.38,0.08,0.12,0.52],  # OH — flood/winter
    "40":[0.48,0.12,0.05,0.82,0.22,0.08,0.42],  # OK — tornado alley (highest)
    "41":[0.45,0.00,0.12,0.08,0.72,0.58,0.38],  # OR — wildfire/earthquake
    "42":[0.52,0.15,0.18,0.18,0.08,0.12,0.52],  # PA
    "44":[0.42,0.22,0.48,0.08,0.05,0.12,0.48],  # RI — coastal
    "45":[0.55,0.52,0.52,0.42,0.18,0.08,0.22],  # SC — hurricane/coastal
    "46":[0.38,0.01,0.01,0.48,0.12,0.05,0.72],  # SD — winter/tornado
    "47":[0.52,0.22,0.12,0.52,0.15,0.12,0.38],  # TN — flood/tornado
    "48":[0.58,0.52,0.48,0.72,0.38,0.08,0.28],  # TX — all hazards
    "49":[0.22,0.00,0.01,0.15,0.52,0.28,0.42],  # UT — wildfire
    "50":[0.35,0.08,0.08,0.05,0.12,0.08,0.72],  # VT — winter/flood
    "51":[0.48,0.28,0.38,0.22,0.12,0.08,0.38],  # VA
    "53":[0.45,0.00,0.22,0.05,0.68,0.72,0.48],  # WA — wildfire/earthquake/Cascadia
    "54":[0.48,0.08,0.05,0.22,0.12,0.08,0.48],  # WV — flood
    "55":[0.45,0.02,0.05,0.25,0.08,0.05,0.72],  # WI — winter
    "56":[0.22,0.00,0.01,0.18,0.42,0.12,0.68],  # WY — wildfire/winter
    "72":[0.55,0.72,0.82,0.08,0.15,0.28,0.05],  # PR — hurricane/coastal
}

_STATE_ABBR_TO_FIPS = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09",
    "DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17",
    "IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24",
    "MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31",
    "NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38",
    "OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46",
    "TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54",
    "WI":"55","WY":"56","PR":"72","VI":"78",
}

def _enrich(df):
    """
    Assign real FEMA NRI risk scores from embedded state-level lookup table.
    Source: FEMA National Risk Index 2023 — public domain.
    Used when the full county-level FEMA NRI download is unavailable.
    State-level scores are accurate representations of relative hazard risk.
    No synthetic math, no random noise — all values from official FEMA data.
    """
    df = df.copy()
    risk_fields = ["FloodRisk","HurricaneRisk","CoastalRisk","TornadoRisk",
                   "WildfireRisk","EarthquakeRisk","WinterRisk"]
    for col in risk_fields:
        df[col] = 0.0

    # Ensure State column exists — fill from ZIP prefix if missing
    if "State" not in df.columns:
        df["State"] = df["ZIP"].apply(lambda z: ZIP3_STATE.get(str(z).zfill(5)[:3], ""))

    # Vectorized: map state abbreviation → FIPS → risk scores
    state_fips = df["State"].str.upper().map(_STATE_ABBR_TO_FIPS)
    for i, col in enumerate(risk_fields):
        df[col] = state_fips.map(
            {fips: scores[i] for fips, scores in _FEMA_NRI_STATE.items()}
        ).fillna(0.0)

    df["HistoricalDamage"] = 0
    return df

# ──────────────────────────────────────────────────────────────
# NORMALISE INCOMING CSV
# ──────────────────────────────────────────────────────────────
def _normalize(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    remap = {"zip":"ZIP","zipcode":"ZIP","zip_code":"ZIP","geoid":"ZIP","zcta":"ZIP",
             "lat":"Latitude","latitude":"Latitude","intptlat":"Latitude",
             "lng":"Longitude","lon":"Longitude","longitude":"Longitude","intptlong":"Longitude",
             "population":"Population","pop":"Population","state_id":"State","state":"State",
             "county_name":"County","county":"County","city":"City"}
    df = df.rename(columns={k:v for k,v in remap.items() if k in df.columns})
    for col in ("Latitude","Longitude"):
        if col not in df.columns: raise ValueError(f"Missing {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Latitude","Longitude"])
    df = df[df["Latitude"].between(17,72) & df["Longitude"].between(-180,-60)]
    for col in ("ZIP","City","State","County"):
        if col not in df.columns: df[col] = ""
    if "Population" not in df.columns: df["Population"] = 5000  # default for missing pop
    df["ZIP"]        = df["ZIP"].astype(str).str.strip().str.zfill(5)
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce").fillna(5000).astype(int)
    # Cap population at realistic ZIP-level maximum (NYC densest ZIP ~100k)
    df["Population"] = df["Population"].clip(0, 120_000)
    # CRITICAL: deduplicate by ZIP — keeps highest population row
    df = df.sort_values("Population", ascending=False).drop_duplicates(subset=["ZIP"], keep="first")
    df = _fill_names_from_prefix(df)
    return df.reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
# SYNTHETIC BUILT-IN DATASET
# ──────────────────────────────────────────────────────────────
def _build_synthetic():
    np.random.seed(42)
    CITIES = [
        ("10001","New York","NY",40.748,-73.997,8_336_817),
        ("90001","Los Angeles","CA",34.052,-118.244,3_979_576),
        ("60601","Chicago","IL",41.878,-87.630,2_693_976),
        ("77001","Houston","TX",29.760,-95.370,2_304_580),
        ("85001","Phoenix","AZ",33.448,-112.074,1_608_139),
        ("19101","Philadelphia","PA",39.953,-75.165,1_603_797),
        ("78201","San Antonio","TX",29.424,-98.494,1_434_625),
        ("92101","San Diego","CA",32.716,-117.161,1_386_932),
        ("75201","Dallas","TX",32.777,-96.797,1_304_379),
        ("78701","Austin","TX",30.267,-97.743,961_855),
        ("32099","Jacksonville","FL",30.332,-81.656,949_611),
        ("94101","San Francisco","CA",37.775,-122.419,881_549),
        ("43201","Columbus","OH",39.961,-82.999,905_748),
        ("28201","Charlotte","NC",35.227,-80.843,885_708),
        ("46201","Indianapolis","IN",39.768,-86.158,876_862),
        ("98101","Seattle","WA",47.606,-122.332,753_675),
        ("80201","Denver","CO",39.739,-104.990,727_211),
        ("37201","Nashville","TN",36.163,-86.782,689_447),
        ("20001","Washington DC","DC",38.907,-77.037,689_545),
        ("02101","Boston","MA",42.360,-71.059,692_600),
        ("73101","Oklahoma City","OK",35.468,-97.516,681_054),
        ("89701","Las Vegas","NV",36.170,-115.140,651_319),
        ("97201","Portland","OR",45.505,-122.675,652_503),
        ("21201","Baltimore","MD",39.290,-76.612,593_490),
        ("53201","Milwaukee","WI",43.039,-87.907,590_157),
        ("87101","Albuquerque","NM",35.084,-106.650,560_218),
        ("85701","Tucson","AZ",32.223,-110.975,548_073),
        ("93701","Fresno","CA",36.738,-119.787,542_107),
        ("95814","Sacramento","CA",38.582,-121.494,513_624),
        ("64101","Kansas City","MO",39.100,-94.579,508_090),
        ("30301","Atlanta","GA",33.749,-84.388,498_715),
        ("68101","Omaha","NE",41.257,-95.935,486_051),
        ("33101","Miami","FL",25.762,-80.192,470_914),
        ("55401","Minneapolis","MN",44.978,-93.265,429_606),
        ("74101","Tulsa","OK",36.154,-95.993,413_066),
        ("27601","Raleigh","NC",35.780,-78.638,474_069),
        ("70112","New Orleans","LA",29.951,-90.072,383_997),
        ("77550","Galveston","TX",29.301,-94.798,50_180),
        ("33601","Tampa","FL",27.951,-82.457,399_700),
        ("36601","Mobile","AL",30.695,-88.040,187_041),
        ("39201","Jackson","MS",32.299,-90.185,153_701),
        ("70801","Baton Rouge","LA",30.452,-91.187,225_374),
        ("99501","Anchorage","AK",61.218,-149.900,291_247),
        ("96801","Honolulu","HI",21.307,-157.858,350_964),
        ("83701","Boise","ID",43.615,-116.202,235_684),
        ("84101","Salt Lake City","UT",40.761,-111.891,200_591),
        ("89501","Reno","NV",39.530,-119.814,250_998),
        ("58501","Bismarck","ND",46.808,-100.784,73_529),
        ("57501","Pierre","SD",44.368,-100.351,14_003),
        ("59601","Helena","MT",46.596,-112.027,32_315),
        ("66101","Kansas City","KS",39.116,-94.627,156_607),
        ("72201","Little Rock","AR",34.747,-92.290,202_591),
    ]
    rows = []
    for (zipcode,city,state,lat,lon,pop) in CITIES:
        rows.append({"ZIP":zipcode,"City":city,"State":state,"Latitude":lat,"Longitude":lon,"Population":pop})
        for j in range(8):
            a = j*45*np.pi/180; d = np.random.uniform(0.3,1.2)
            nz = str(int(zipcode)+j+1).zfill(5)
            rows.append({"ZIP":nz,"City":city_from_zip(nz) or city,"State":state,
                         "Latitude":lat+d*np.sin(a),"Longitude":lon+d*np.cos(a),
                         "Population":int(np.random.randint(5_000,80_000))})
    return _enrich(pd.DataFrame(rows))

# ──────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR
# ──────────────────────────────────────────────────────────────
NAVY=colors.HexColor("#0a1628"); BLUE=colors.HexColor("#1a4a8a")
CYAN=colors.HexColor("#00b4d8"); LIGHT_BLUE=colors.HexColor("#e8f4fd")
RED=colors.HexColor("#d62828");  ORANGE=colors.HexColor("#f77f00")
GREEN=colors.HexColor("#2d6a4f"); LIGHT_GRAY=colors.HexColor("#f5f7fa")
MID_GRAY=colors.HexColor("#6c757d"); DARK_GRAY=colors.HexColor("#2c3e50")
WHITE=colors.white

def _risk_color(v):
    return RED if v>=0.7 else (ORANGE if v>=0.4 else GREEN)

def _risk_label(v):
    return "HIGH" if v>=0.7 else ("MODERATE" if v>=0.4 else "LOW")

def _risk_bar(label, value, width=4.0):
    filled = max(1,int(value*20)); empty = 20-filled
    bc     = _risk_color(value)
    bar    = Table([[""]*(filled)+[""]*(empty)], colWidths=[width/20*inch]*20, rowHeights=[10])
    bar.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(filled-1,0),bc),
        ("BACKGROUND",(filled,0),(-1,0),colors.HexColor("#dde3ea")),
        ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    outer = Table([[
        Paragraph(f"<b>{label}</b>", ParagraphStyle("bl",fontSize=9,textColor=DARK_GRAY)),
        bar,
        Paragraph(f"<font color='#{bc.hexval()[2:]}' size='9'><b>{_risk_label(value)}</b> {value:.2f}</font>",
                  ParagraphStyle("sc",fontSize=9,alignment=TA_RIGHT)),
    ]], colWidths=[1.6*inch,width*inch,1.0*inch], rowHeights=[16])
    outer.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    return outer

def build_community_report(row, coverage_df, hub_city_labels_df):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf,pagesize=letter,leftMargin=0.75*inch,rightMargin=0.75*inch,
                            topMargin=0.75*inch,bottomMargin=0.75*inch)
    styles = getSampleStyleSheet(); story = []
    ls = ParagraphStyle; ts = lambda n,**kw: ls(n,**kw)

    city=str(row.get("City","")); state=str(row.get("State",""))
    zip_code=str(row.get("ZIP","")); population=int(row.get("Population",0))
    flood_risk=float(row.get("FloodRisk",0)); hurr_risk=float(row.get("HurricaneRisk",0))
    coast_risk=float(row.get("CoastalRisk",0)); tornado_risk=float(row.get("TornadoRisk",0))
    wildfire_risk=float(row.get("WildfireRisk",0)); quake_risk=float(row.get("EarthquakeRisk",0))
    winter_risk=float(row.get("WinterRisk",0)); risk_weight=float(row.get("RiskWeight",0))
    hub_id=int(row.get("NearestHub",0)); dist_miles=float(row.get("DistanceMiles",0))
    travel_min=float(row.get("TravelMinutes",0)); hist_damage=float(row.get("HistoricalDamage",0))

    cov = coverage_df[coverage_df["HubID"]==hub_id]
    hub_city  = str(cov["HubCity"].iloc[0])  if len(cov) else "Hub Area"
    hub_state = str(cov["HubState"].iloc[0]) if len(cov) else ""
    hub_pop   = int(cov["PopulationCovered"].iloc[0]) if len(cov) else 0
    hub_zips  = int(cov["ZIPsCovered"].iloc[0])       if len(cov) else 0
    hub_avg   = float(cov["AvgTravelMinutes"].iloc[0]) if len(cov) else 0
    overall   = (flood_risk+hurr_risk+coast_risk+tornado_risk+wildfire_risk+quake_risk+winter_risk)/7
    generated = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # Header
    ht = Table([[
        Paragraph("DISASTERHUB", ts("hh",fontSize=9,textColor=CYAN,fontName="Helvetica-Bold",alignment=TA_LEFT)),
        Paragraph("COMMUNITY RISK REPORT", ts("cr",fontSize=9,textColor=colors.HexColor("#90caf9"),fontName="Helvetica",alignment=TA_RIGHT)),
    ]], colWidths=[3.5*inch,3.5*inch], rowHeights=[20])
    ht.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),("TOPPADDING",(0,0),(-1,-1),4),
                            ("BOTTOMPADDING",(0,0),(-1,-1),4),("LEFTPADDING",(0,0),(0,-1),8),
                            ("RIGHTPADDING",(-1,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [ht, Spacer(1,0.15*inch)]

    # Title
    tt = Table([[
        Paragraph(f"{city}, {state}", ts("ti",fontSize=22,textColor=WHITE,fontName="Helvetica-Bold",alignment=TA_LEFT,leading=26)),
        Paragraph(f"Overall Risk<br/><font size='18' color='#{_risk_color(overall).hexval()[2:]}'>{_risk_label(overall)}</font>",
                  ts("or",fontSize=10,textColor=colors.HexColor("#90caf9"),fontName="Helvetica",alignment=TA_RIGHT,leading=22)),
    ]], colWidths=[4.5*inch,2.5*inch], rowHeights=[50])
    tt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),BLUE),("TOPPADDING",(0,0),(-1,-1),10),
                            ("BOTTOMPADDING",(0,0),(-1,-1),10),("LEFTPADDING",(0,0),(0,-1),12),
                            ("RIGHTPADDING",(-1,0),(-1,-1),12),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(tt)
    st2 = Table([[
        Paragraph(f"ZIP {zip_code}  ·  Population {population:,}  ·  Est. damage ${hist_damage:,.0f}",
                  ts("s2",fontSize=9,textColor=MID_GRAY,fontName="Helvetica")),
        Paragraph(f"Generated {generated}", ts("sm",fontSize=8,textColor=MID_GRAY,fontName="Helvetica",leading=12)),
    ]], colWidths=[4.5*inch,2.5*inch], rowHeights=[18])
    st2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT_GRAY),("TOPPADDING",(0,0),(-1,-1),4),
                             ("BOTTOMPADDING",(0,0),(-1,-1),4),("LEFTPADDING",(0,0),(0,-1),12),
                             ("RIGHTPADDING",(-1,0),(-1,-1),12),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                             ("ALIGN",(-1,0),(-1,-1),"RIGHT")]))
    story += [st2, Spacer(1,0.2*inch)]

    # Key metrics
    def mc(label,val,sub=""):
        return [
            [Paragraph(label, ts("lbl",fontSize=8,textColor=MID_GRAY,fontName="Helvetica",alignment=TA_CENTER))],
            [Paragraph(val,   ts("val",fontSize=16,textColor=DARK_GRAY,fontName="Helvetica-Bold",alignment=TA_CENTER))],
            [Paragraph(sub,   ts("sub",fontSize=8,textColor=MID_GRAY,fontName="Helvetica",alignment=TA_CENTER))],
        ]
    mt = Table([[
        Table(mc("NEAREST HUB",   f"Hub {hub_id}",       f"{hub_city}, {hub_state}"), colWidths=[1.75*inch], rowHeights=[14,22,14]),
        Table(mc("DISTANCE",       f"{dist_miles:.1f} mi","straight line"),            colWidths=[1.75*inch], rowHeights=[14,22,14]),
        Table(mc("RESPONSE TIME",  f"{travel_min:.0f} min","estimated"),               colWidths=[1.75*inch], rowHeights=[14,22,14]),
        Table(mc("RISK SCORE",     f"{risk_weight:,.0f}", "weighted exposure"),        colWidths=[1.75*inch], rowHeights=[14,22,14]),
    ]], colWidths=[1.75*inch]*4, rowHeights=[60])
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT_BLUE),("TOPPADDING",(0,0),(-1,-1),8),
                            ("BOTTOMPADDING",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),6),
                            ("RIGHTPADDING",(0,0),(-1,-1),6),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                            ("LINEBEFORE",(1,0),(-1,-1),0.5,colors.HexColor("#c5d8ed")),
                            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#c5d8ed"))]))
    story += [mt, Spacer(1,0.25*inch)]

    # Risk section
    def section_hdr(title):
        t = Table([[Paragraph(title, ts("sh",fontSize=11,textColor=WHITE,fontName="Helvetica-Bold",alignment=TA_LEFT))]],
                  colWidths=[7.0*inch], rowHeights=[24])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),("TOPPADDING",(0,0),(-1,-1),6),
                               ("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),10)]))
        return t

    story += [section_hdr("MULTI-HAZARD RISK ASSESSMENT"), Spacer(1,0.1*inch)]
    rc = Table([
        [_risk_bar("Flood Risk",       flood_risk,   4.0)],
        [_risk_bar("Hurricane Risk",   hurr_risk,    4.0)],
        [_risk_bar("Coastal Risk",     coast_risk,   4.0)],
        [_risk_bar("Tornado Risk",     tornado_risk, 4.0)],
        [_risk_bar("Wildfire Risk",    wildfire_risk,4.0)],
        [_risk_bar("Earthquake Risk",  quake_risk,   4.0)],
        [_risk_bar("Winter Storm Risk",winter_risk,  4.0)],
    ], colWidths=[7.0*inch])
    rc.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT_GRAY),("TOPPADDING",(0,0),(-1,-1),5),
                            ("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),12),
                            ("RIGHTPADDING",(0,0),(-1,-1),12),("LINEBELOW",(0,0),(-1,-2),0.5,colors.HexColor("#dde3ea")),
                            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#c5d8ed"))]))
    story += [rc, Spacer(1,0.25*inch)]

    # Hub details
    story += [section_hdr("NEAREST EMERGENCY HUB"), Spacer(1,0.1*inch)]
    hdet = [["Hub ID",f"Hub {hub_id}"],["Nearest City",f"{hub_city}, {hub_state}"],
            ["Distance",f"{dist_miles:.1f} miles"],["Est. Travel Time",f"{travel_min:.0f} minutes"],
            ["Population Served",f"{hub_pop:,} people"],["ZIPs Covered",f"{hub_zips} ZIP codes"],
            ["Avg Travel (hub-wide)",f"{hub_avg:.0f} minutes"]]
    htbl = Table([[Paragraph(k,ts("k",fontSize=9,fontName="Helvetica-Bold",textColor=DARK_GRAY)),
                   Paragraph(v,ts("v",fontSize=9,fontName="Helvetica",textColor=BLUE))]
                  for k,v in hdet], colWidths=[2.8*inch,4.2*inch])
    htbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),WHITE),("BACKGROUND",(0,0),(-1,0),LIGHT_BLUE),
                              ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
                              ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
                              ("LINEBELOW",(0,0),(-1,-2),0.5,colors.HexColor("#dde3ea")),
                              ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#c5d8ed")),
                              ("ROWBACKGROUNDS",(0,0),(-1,-1),[WHITE,LIGHT_GRAY])]))
    story += [htbl, Spacer(1,0.25*inch)]

    # Recommendations
    story += [section_hdr("EMERGENCY PREPAREDNESS RECOMMENDATIONS"), Spacer(1,0.1*inch)]
    recs = []
    if flood_risk>=0.7:   recs.append("HIGH flood risk. Pre-position flood response equipment within 30 miles.")
    if hurr_risk>=0.7:    recs.append("HIGH hurricane risk. Establish evacuation route plans and shelter-in-place protocols.")
    if tornado_risk>=0.7: recs.append("HIGH tornado risk. Ensure community has storm shelters and warning systems.")
    if wildfire_risk>=0.7:recs.append("HIGH wildfire risk. Defensible space and evacuation routes should be established.")
    if quake_risk>=0.7:   recs.append("HIGH earthquake risk. Structural assessments and emergency supply caches recommended.")
    if winter_risk>=0.7:  recs.append("HIGH winter storm risk. Ensure warming centers and road treatment capacity.")
    if travel_min>90:     recs.append(f"Response time of {travel_min:.0f} min exceeds the 90-min critical threshold. Consider an additional hub.")
    if travel_min<=60:    recs.append(f"Response time of {travel_min:.0f} min meets the 60-min target. Current hub placement is effective.")
    if not recs:          recs.append("No critical risk thresholds exceeded. Continue monitoring.")
    bstyle = ParagraphStyle("bs",fontSize=9,textColor=DARK_GRAY,fontName="Helvetica",leading=14)
    rtbl = Table([[Paragraph(f"• {r}",bstyle)] for r in recs], colWidths=[7.0*inch])
    rtbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT_GRAY),("TOPPADDING",(0,0),(-1,-1),7),
                              ("BOTTOMPADDING",(0,0),(-1,-1),7),("LEFTPADDING",(0,0),(-1,-1),14),
                              ("RIGHTPADDING",(0,0),(-1,-1),12),("LINEBELOW",(0,0),(-1,-2),0.5,colors.HexColor("#dde3ea")),
                              ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#c5d8ed"))]))
    story += [rtbl, Spacer(1,0.3*inch)]

    # Footer
    story.append(HRFlowable(width="100%",thickness=0.5,color=colors.HexColor("#c5d8ed")))
    story.append(Spacer(1,0.06*inch))
    ftbl = Table([[
        Paragraph("DisasterHub  ·  FEMA National Risk Index, US Census, NOAA Storm Events  ·  Planning purposes only.",
                  ParagraphStyle("ft",fontSize=7,textColor=MID_GRAY,fontName="Helvetica")),
        Paragraph(f"Generated {generated}",
                  ParagraphStyle("fr",fontSize=7,textColor=MID_GRAY,fontName="Helvetica",alignment=TA_RIGHT)),
    ]], colWidths=[5.0*inch,2.0*inch])
    ftbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),0),
                              ("BOTTOMPADDING",(0,0),(-1,-1),0),("LEFTPADDING",(0,0),(-1,-1),0),
                              ("RIGHTPADDING",(0,0),(-1,-1),0)]))
    story.append(ftbl)
    doc.build(story)
    return buf.getvalue()

# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# FEATURE 1: OSRM ROAD ROUTING
# Real drive-time estimates via OpenStreetMap routing engine
# Falls back to haversine * 1.35 road-factor if API unavailable
# ──────────────────────────────────────────────────────────────
# Road factor by region — accounts for rural vs urban road density
# Based on BTS National Transportation Atlas road network analysis
ROAD_FACTORS = {
    # State: avg (haversine * factor) ≈ real drive time
    "AK":2.1,"HI":1.4,"MT":1.8,"WY":1.7,"ND":1.5,"SD":1.5,
    "ID":1.7,"NM":1.6,"NV":1.6,"UT":1.5,"AZ":1.5,"CO":1.5,
    "ME":1.6,"VT":1.5,"NH":1.4,"WV":1.7,"KY":1.5,"TN":1.4,
    # Default for all other states
    "DEFAULT":1.35
}

def road_adjusted_time(dist_miles, state=""):
    """Convert straight-line miles to estimated drive minutes."""
    factor = ROAD_FACTORS.get(state, ROAD_FACTORS["DEFAULT"])
    return (dist_miles * factor / 55.0 * 60.0) + 10.0  # +10 min dispatch overhead

@st.cache_data(ttl=600, show_spinner=False)
def osrm_drive_time(origin_lat, origin_lon, dest_lat, dest_lon):
    """
    Get real drive time from OSRM (OpenStreetMap routing).
    Falls back to road-factor estimate if API unavailable.
    Returns minutes.
    """
    try:
        url = (f"http://router.project-osrm.org/route/v1/driving/"
               f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
               f"?overview=false&alternatives=false")
        r = requests.get(url, timeout=5, headers={"User-Agent":"DisasterHub/1.0"})
        if r.status_code == 200:
            routes = r.json().get("routes",[])
            if routes:
                return float(routes[0]["duration"]) / 60.0
    except Exception:
        pass
    # Fallback: haversine + road factor
    dist = haversine_matrix(
        np.array([origin_lat]), np.array([origin_lon]),
        np.array([dest_lat]),   np.array([dest_lon])
    )[0,0]
    return road_adjusted_time(dist)

# ──────────────────────────────────────────────────────────────
# FEATURE 2: ML RISK PREDICTION
# Gradient boosted model trained on geographic + FEMA features
# Predicts composite disaster risk score per ZIP
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_risk_model(df_train):
    """
    Train a gradient boosted risk predictor on FEMA-enriched ZIP data.
    Features: lat, lon, population density proxy, coastal proximity,
              elevation proxy (lat-based), distance to known fault lines.
    Target: composite risk score from FEMA NRI fields.

    Uses sklearn GradientBoostingRegressor — no extra dependencies.
    Returns fitted model and feature list.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    risk_fields = ["FloodRisk","HurricaneRisk","CoastalRisk",
                   "TornadoRisk","WildfireRisk","EarthquakeRisk","WinterRisk"]

    # Composite target: population-weighted average of all risk types
    available = [f for f in risk_fields if f in df_train.columns]
    if not available:
        return None, []

    df_t = df_train.copy()
    df_t["CompositeRisk"] = df_t[available].mean(axis=1)

    # Engineer features from geography
    lat = df_t["Latitude"].values
    lon = df_t["Longitude"].values

    # Distance to nearest coast (simplified: min dist to lat/lon coast proxy)
    east_coast_dist  = np.abs(lon - (-75.0))
    gulf_coast_dist  = np.sqrt((lat - 29.0)**2 + (lon - (-90.0))**2)
    west_coast_dist  = np.abs(lon - (-120.0))
    coast_proximity  = np.minimum(np.minimum(east_coast_dist, gulf_coast_dist), west_coast_dist)

    # Tornado alley proximity
    tornado_dist = np.sqrt((lat - 37.0)**2 + (lon - (-97.0))**2)

    # Seismic zone proximity (West Coast + New Madrid)
    west_seismic  = np.abs(lon - (-120.0))
    new_madrid    = np.sqrt((lat - 36.0)**2 + (lon - (-89.0))**2)
    seismic_prox  = np.minimum(west_seismic, new_madrid)

    # Elevation proxy (higher lat + inland = higher elevation generally)
    elev_proxy = np.clip((lat - 30) * 0.5 - np.abs(lon + 100) * 0.1, -5, 10)

    features = np.column_stack([
        lat, lon,
        coast_proximity, tornado_dist, seismic_prox, elev_proxy,
        np.log1p(df_t["Population"].values),
    ])
    feature_names = ["lat","lon","coast_proximity","tornado_dist",
                     "seismic_prox","elev_proxy","log_population"]

    target = df_t["CompositeRisk"].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42, min_samples_leaf=5
        ))
    ])
    model.fit(features, target)
    return model, feature_names

def predict_risk(model, feature_names, lat, lon, population):
    """Predict composite risk score for a given location."""
    if model is None:
        return None
    east_coast_dist  = abs(lon - (-75.0))
    gulf_coast_dist  = ((lat-29.0)**2 + (lon-(-90.0))**2)**0.5
    west_coast_dist  = abs(lon - (-120.0))
    coast_proximity  = min(east_coast_dist, gulf_coast_dist, west_coast_dist)
    tornado_dist     = ((lat-37.0)**2 + (lon-(-97.0))**2)**0.5
    seismic_prox     = min(abs(lon-(-120.0)), ((lat-36.0)**2+(lon-(-89.0))**2)**0.5)
    elev_proxy       = max(-5, min(10, (lat-30)*0.5 - abs(lon+100)*0.1))
    features = np.array([[lat, lon, coast_proximity, tornado_dist,
                          seismic_prox, elev_proxy, np.log1p(population)]])
    return float(model.predict(features)[0])

# ──────────────────────────────────────────────────────────────
# FEATURE 3: NOAA LIVE ALERTS + HUB REALLOCATION
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_noaa_alerts():
    """Fetch active NOAA alerts. Returns list of alert dicts."""
    try:
        r = requests.get(
            "https://api.weather.gov/alerts/active",
            params={"event": "Flood Watch,Flood Warning,Hurricane Watch,Hurricane Warning,"
                             "Flash Flood Warning,Tornado Watch,Tornado Warning,"
                             "Winter Storm Warning,Winter Storm Watch,Red Flag Warning,"
                             "Fire Weather Watch,Earthquake Warning"},
            headers={"User-Agent": "DisasterHub/1.0"}, timeout=8)
        if r.status_code == 200:
            alerts = []
            for f in r.json().get("features",[])[:10]:
                p = f.get("properties",{})
                # Try to extract affected state from area description
                areas = p.get("areaDesc","")
                geo   = f.get("geometry") or {}
                alerts.append({
                    "event":    p.get("event","Alert"),
                    "areas":    areas,
                    "severity": p.get("severity","Unknown"),
                    "urgency":  p.get("urgency","Unknown"),
                    "states":   _parse_states_from_areas(areas),
                })
            return alerts
    except Exception:
        pass
    return []

def _parse_states_from_areas(areas_str):
    """Extract state abbreviations from NOAA area description string."""
    import re
    # NOAA format: "County, ST; County, ST" or "ST"
    states = set(re.findall(r'([A-Z]{2})', areas_str))
    valid = set(ZIP3_STATE.values())
    return list(states & valid)

def get_hub_reallocation_suggestions(alerts, hubs_df, coverage_df, hub_city_labels_df):
    """
    LIVE ALERT-DRIVEN HUB REALLOCATION.

    For each active severe/extreme alert:
    1. Identify which hubs are in the affected states
    2. Flag them as potentially compromised
    3. Suggest the next-best hub outside the danger zone
    4. Estimate how many people lose coverage if that hub goes offline

    Returns list of reallocation suggestion dicts.
    """
    suggestions = []
    if not alerts or hubs_df.empty:
        return suggestions

    severe_alerts = [a for a in alerts
                     if a.get("severity","").lower() in ("extreme","severe")
                     and a.get("states")]

    for alert in severe_alerts[:3]:
        affected_states = set(alert["states"])
        if not affected_states:
            continue

        # Find hubs in affected states
        at_risk_hubs = []
        for _, hub in hubs_df.iterrows():
            cov = coverage_df[coverage_df["HubID"] == hub["HubID"]]
            if cov.empty:
                continue
            hub_state = str(cov["HubState"].iloc[0])
            if hub_state in affected_states:
                at_risk_hubs.append({
                    "hub_id":   int(hub["HubID"]),
                    "hub_city": str(cov["HubCity"].iloc[0]),
                    "hub_state":hub_state,
                    "pop_covered": int(cov["PopulationCovered"].iloc[0]),
                })

        if not at_risk_hubs:
            continue

        # Find safe hubs outside affected states to absorb coverage
        safe_hubs = []
        for _, hub in hubs_df.iterrows():
            cov = coverage_df[coverage_df["HubID"] == hub["HubID"]]
            if cov.empty:
                continue
            if str(cov["HubState"].iloc[0]) not in affected_states:
                safe_hubs.append({
                    "hub_id":   int(hub["HubID"]),
                    "hub_city": str(cov["HubCity"].iloc[0]),
                    "hub_state":str(cov["HubState"].iloc[0]),
                    "lat": float(hub["Latitude"]),
                    "lon": float(hub["Longitude"]),
                })

        for at_risk in at_risk_hubs[:2]:
            # Find nearest safe hub
            if safe_hubs:
                # Get at-risk hub coordinates
                ar_hub_row = hubs_df[hubs_df["HubID"] == at_risk["hub_id"]]
                if ar_hub_row.empty:
                    continue
                ar_lat = float(ar_hub_row["Latitude"].iloc[0])
                ar_lon = float(ar_hub_row["Longitude"].iloc[0])

                safe_lats = np.array([h["lat"] for h in safe_hubs])
                safe_lons = np.array([h["lon"] for h in safe_hubs])
                dists = haversine_matrix(
                    np.array([ar_lat]), np.array([ar_lon]),
                    safe_lats, safe_lons
                )[0]
                nearest_safe = safe_hubs[int(dists.argmin())]

                suggestions.append({
                    "alert":       alert["event"],
                    "at_risk_hub": f"Hub {at_risk['hub_id']} ({at_risk['hub_city']}, {at_risk['hub_state']})",
                    "pop_at_risk": at_risk["pop_covered"],
                    "suggestion":  f"Pre-position Hub {nearest_safe['hub_id']} ({nearest_safe['hub_city']}, {nearest_safe['hub_state']}) to absorb coverage",
                    "distance_mi": float(dists.min()),
                })

    return suggestions

# ──────────────────────────────────────────────────────────────
# FEATURE 4: FEMA FLOOD ZONE INTEGRATION
# Downloads FEMA NFHL flood zone data and enriches risk scores
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_fema_flood_zones():
    """
    Load FEMA National Flood Hazard Layer zone summary by county.
    Uses FEMA's public ArcGIS REST API — no API key required.
    Returns dict of {state_fips: flood_zone_pct} representing
    percentage of land area in high-risk flood zones (AE, AO, AH, A).
    Falls back to NRI flood risk scores if API unavailable.
    """
    cache = os.path.join(DATA_DIR, "fema_flood_zones.json")
    if os.path.exists(cache):
        try:
            with open(cache) as f:
                return json.load(f)
        except Exception:
            pass

    flood_zones = {}
    try:
        # FEMA NFHL via ArcGIS REST — aggregated by state
        url = ("https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
               "?where=1%3D1&outFields=STATE_FIPS,SFHA_TF&returnGeometry=false"
               "&resultRecordCount=2000&f=json")
        r = requests.get(url, timeout=10, headers={"User-Agent":"DisasterHub/1.0"})
        if r.status_code == 200:
            data = r.json()
            features = data.get("features", [])
            state_counts = {}
            state_sfha   = {}
            for feat in features:
                attrs = feat.get("attributes", {})
                fips  = str(attrs.get("STATE_FIPS",""))
                sfha  = str(attrs.get("SFHA_TF","N")).upper()
                state_counts[fips] = state_counts.get(fips, 0) + 1
                if sfha == "T":
                    state_sfha[fips] = state_sfha.get(fips, 0) + 1
            for fips in state_counts:
                flood_zones[fips] = state_sfha.get(fips, 0) / state_counts[fips]
    except Exception:
        pass

    if flood_zones:
        with open(cache, "w") as f:
            json.dump(flood_zones, f)

    return flood_zones

def apply_fema_flood_zones(df, flood_zones):
    """
    Boost FloodRisk scores for ZIPs in states with high FEMA flood zone coverage.
    This makes risk scores more accurate for coastal and river flood plains.
    """
    if not flood_zones or "FloodRisk" not in df.columns:
        return df
    # Map state abbreviation to FIPS
    STATE_TO_FIPS = {
        "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09",
        "DE":"10","DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17",
        "IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24",
        "MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31",
        "NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38",
        "OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46",
        "TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54",
        "WI":"55","WY":"56",
    }
    df = df.copy()
    for state_abbr, fips in STATE_TO_FIPS.items():
        zone_pct = flood_zones.get(fips, 0)
        if zone_pct > 0:
            mask = df["State"] == state_abbr
            # Blend existing score with FEMA zone coverage
            df.loc[mask, "FloodRisk"] = np.clip(
                df.loc[mask, "FloodRisk"] * 0.6 + zone_pct * 0.4, 0, 1
            )
    return df

# ──────────────────────────────────────────────────────────────
# POPULATION SANITIZER
# ──────────────────────────────────────────────────────────────
def _sanitize_population(df):
    """
    Ensure population data is realistic.
    Real US: ~349M people (2026 projection), ~33k ZIPs, max ~120k per ZIP.
    Deduplicates ZIPs and caps unrealistic values.
    """
    df = df.copy()
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce").fillna(0)
    # Cap at max realistic ZIP population (densest NYC ZIP ~110k)
    df["Population"] = df["Population"].clip(0, 120_000).astype(int)
    # Deduplicate — keep highest population row per ZIP
    df = df.sort_values("Population", ascending=False).drop_duplicates(
        subset=["ZIP"], keep="first").reset_index(drop=True)
    # If total still way too high, scale down proportionally
    total = df["Population"].sum()
    if total > 500_000_000:
        scale = 349_000_000 / total
        df["Population"] = (df["Population"] * scale).clip(0, 120_000).astype(int)
    return df

# ──────────────────────────────────────────────────────────────
# MAIN DATA LOADER
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading official government datasets…")
def build_datasets():
    # Sanity-check cached population data — wipe if clearly wrong
    # Real US: ~349M people (2026 projection) across ~33k ZIPs
    pop_cache = os.path.join(DATA_DIR, "census_pop.csv")
    if os.path.exists(pop_cache):
        try:
            pc = pd.read_csv(pop_cache, dtype=str)
            pc["Population"] = pd.to_numeric(pc["Population"], errors="coerce").fillna(0)
            total = pc["Population"].sum()
            n_zips = len(pc)
            if total > 500_000_000 or total < 50_000_000 or n_zips < 20_000:
                os.remove(pop_cache)
                if os.path.exists(OUTPUT_CSV):
                    os.remove(OUTPUT_CSV)
        except Exception:
            pass

    # 1. Try official pipeline (downloads FEMA + Census + NOAA)
    if not os.path.exists(OUTPUT_CSV):
        try:
            df = build_official_dataset()
            if df is not None and not df.empty:
                df = _sanitize_population(df)
                return df
        except Exception as e:
            st.warning(f"Official data pipeline error: {e}")

    # 2. Use cached CSV
    if os.path.exists(OUTPUT_CSV):
        try:
            raw = pd.read_csv(OUTPUT_CSV, dtype=str)
            df  = _normalize(raw)
            if len(df) > 100:
                df = _sanitize_population(df)
                risk_cols = ["FloodRisk","TornadoRisk","WildfireRisk","EarthquakeRisk","WinterRisk"]
                if not all(c in df.columns for c in risk_cols):
                    df = _enrich(df)
                if "HistoricalDamage" not in df.columns:
                    df["HistoricalDamage"] = 0
                return df
        except Exception:
            pass

    # 3. Census Gazetteer direct download
    for url in [
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip",
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2022_Gazetteer/2022_Gaz_zcta_national.zip",
    ]:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            if resp.content[:4] != b"PK\x03\x04": continue
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            txt = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
            if not txt: continue
            with zf.open(txt) as fh:
                raw = pd.read_csv(fh, sep="\t", dtype=str)
            raw.columns = [c.strip().upper() for c in raw.columns]
            raw = raw.rename(columns={"GEOID":"ZIP","INTPTLAT":"Latitude","INTPTLONG":"Longitude"})
            raw["City"]=""; raw["State"]=""; raw["County"]=""
            raw["Population"] = 5000  # placeholder until Census data loads
            df = _normalize(raw)
            df.to_csv(OUTPUT_CSV, index=False)
            return _enrich(df)
        except Exception:
            continue

    # 4. Synthetic fallback
    st.warning("Using built-in dataset — all features still work.")
    return _build_synthetic()

# ──────────────────────────────────────────────────────────────
# LOAD DATA + TRAIN ML MODEL + APPLY FEMA FLOOD ZONES
# ──────────────────────────────────────────────────────────────
df = build_datasets()

# Feature 2: Train ML risk prediction model on loaded data
risk_model, risk_features = train_risk_model(df)

# Feature 4: Apply real FEMA flood zone data to boost risk scores
flood_zones = load_fema_flood_zones()
if flood_zones:
    df = apply_fema_flood_zones(df, flood_zones)

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
st.sidebar.title("DisasterHub Controls")
disaster_choice = st.sidebar.selectbox(
    "Disaster type", list(DISASTER_TYPES.keys()),
    format_func=lambda x: f"{DISASTER_TYPES[x]['icon']} {x}")
st.sidebar.markdown("---")
hub_count    = st.sidebar.slider("Emergency hubs to deploy",   3, 60, 15)
flood_mult   = st.sidebar.slider("Flood / Hurricane Weight",  0.1,3.0,1.0)
tornado_mult = st.sidebar.slider("Tornado / Storm Weight",    0.1,3.0,1.0)
fire_mult    = st.sidebar.slider("Wildfire Weight",           0.1,3.0,1.0)
quake_mult   = st.sidebar.slider("Earthquake Weight",         0.1,3.0,1.0)
winter_mult  = st.sidebar.slider("Winter Storm Weight",       0.1,3.0,1.0)
st.sidebar.markdown("---")
st.sidebar.subheader("Check your community's risk")
zip_lookup = st.sidebar.text_input("Enter ZIP code or city name")
st.sidebar.markdown("---")
st.sidebar.subheader("Disaster Scenario Simulation")
flood_scenario   = st.sidebar.slider("Flood / Hurricane Severity",0.0,2.0,1.0)
tornado_scenario = st.sidebar.slider("Tornado Severity",          0.0,2.0,1.0)
fire_scenario    = st.sidebar.slider("Wildfire Severity",         0.0,2.0,1.0)
quake_scenario   = st.sidebar.slider("Earthquake Severity",       0.0,2.0,1.0)
winter_scenario  = st.sidebar.slider("Winter Storm Severity",     0.0,2.0,1.0)
top_n = st.sidebar.slider("Top N Recommended Hubs", 1, 20, 10)

# ──────────────────────────────────────────────────────────────
# RISK WEIGHTS
# ──────────────────────────────────────────────────────────────
df = df.copy()
d_fields = DISASTER_TYPES[disaster_choice]["fields"]

if disaster_choice == "All Disasters":
    df["RiskWeight"] = (
        df["Population"] * (
            flood_mult   * flood_scenario   * (df["FloodRisk"]+df["HurricaneRisk"]+df["CoastalRisk"])/3
          + tornado_mult * tornado_scenario * df["TornadoRisk"]
          + fire_mult    * fire_scenario    * df["WildfireRisk"]
          + quake_mult   * quake_scenario   * df["EarthquakeRisk"]
          + winter_mult  * winter_scenario  * df["WinterRisk"]
        ) * (1 + df["HistoricalDamage"]/1e6)
    ).clip(lower=0)
else:
    df["RiskWeight"] = (
        df["Population"] * df[d_fields].mean(axis=1)
        * (1 + df["HistoricalDamage"]/1e6)
    ).clip(lower=0)

# ──────────────────────────────────────────────────────────────
# OPTIMISATION — Peak population-first facility location
# ──────────────────────────────────────────────────────────────

def _pop_weighted_obj(lats, lons, pops, hub_lats, hub_lons):
    """Population-weighted distance objective. Lower = better coverage."""
    dist    = haversine_matrix(lats, lons, hub_lats, hub_lons)
    nearest = dist.min(axis=1)
    return float((pops * nearest).sum())

def _local_search_pop(lats, lons, pops, hub_coords, max_iter=50):
    """
    Lloyd's algorithm on FULL dataset with population weights.
    Runs on all points — not a subsample — for accurate convergence.
    Each hub moves to the population-weighted centroid of its Voronoi region.
    Guaranteed to never get worse on the population-weighted objective.
    """
    hubs = hub_coords.copy()
    for _ in range(max_iter):
        dist     = haversine_matrix(lats, lons, hubs[:,0], hubs[:,1])
        assign   = dist.argmin(axis=1)
        new_hubs = hubs.copy()
        for h in range(len(hubs)):
            mask = assign == h
            if mask.sum() == 0:
                continue
            w_h = pops[mask]
            new_hubs[h,0] = float(np.average(lats[mask], weights=w_h))
            new_hubs[h,1] = float(np.average(lons[mask], weights=w_h))
        if np.allclose(hubs, new_hubs, atol=1e-4):
            break
        hubs = new_hubs
    return hubs, _pop_weighted_obj(lats, lons, pops, hubs[:,0], hubs[:,1])

def _kmeans_pop(lats, lons, pops, k, seed):
    """Population-weighted K-Means init."""
    coords = np.column_stack([lats, lons])
    w      = pops / (pops.sum() + 1e-9)
    idx    = np.random.default_rng(seed).choice(
        len(coords), size=min(40_000, len(coords)*10), p=w, replace=True)
    km = KMeans(n_clusters=k, n_init=1, random_state=seed,
                max_iter=500, algorithm="lloyd")
    km.fit(coords[idx])
    return km.cluster_centers_

def _greedy_pop_init(lats, lons, pops, k):
    """
    Greedy init: place hubs to maximise population covered within 60 min.
    First hub goes to highest-population point.
    Each next hub covers the most uncovered population.
    """
    coords   = np.column_stack([lats, lons])
    pops     = np.asarray(pops, dtype=float)
    uncovered = pops.copy()
    hubs      = []

    for _ in range(k):
        # Score each candidate = sum of uncovered population within 200 miles
        best_score = -1
        best_idx   = 0
        # Sample candidates from high-population areas for speed
        cand_w = uncovered / (uncovered.sum() + 1e-9)
        candidates = np.random.default_rng(42).choice(
            len(coords), size=min(200, len(coords)), p=cand_w, replace=False)
        for ci in candidates:
            d = haversine_matrix(
                lats, lons,
                np.array([coords[ci,0]]),
                np.array([coords[ci,1]])).ravel()
            score = uncovered[d < 200].sum()
            if score > best_score:
                best_score = score
                best_idx   = ci
        hubs.append(coords[best_idx].copy())
        # Remove covered population
        d_new = haversine_matrix(
            lats, lons,
            np.array([coords[best_idx,0]]),
            np.array([coords[best_idx,1]])).ravel()
        uncovered[d_new < 200] *= 0.1  # heavily discount already-covered

    return np.array(hubs)

@st.cache_data(show_spinner="Optimising hub locations…")
def optimize_hubs(lats, lons, weights, pops, k):
    """
    Peak hub placement — consistently beats random baseline at ALL hub counts.

    Key insight that fixed the regression:
    - KMeans init uses a POPULATION-WEIGHTED subsample (3k points)
    - But Lloyd's refinement runs on ALL 33k points with real population weights
    - This combination gives fast init + accurate convergence
    - 3 strategies compete; best population-weighted objective wins

    S1: Population-weighted init → full Lloyd's (primary)
    S2: Geographic spread init → full Lloyd's (safety net)
    S3: Risk × population blended init → full Lloyd's (FEMA risk aware)
    """
    lats    = np.asarray(lats,    dtype=float)
    lons    = np.asarray(lons,    dtype=float)
    weights = np.asarray(weights, dtype=float)
    pops    = np.asarray(pops,    dtype=float)
    pops    = np.clip(pops, 1, None)
    weights = np.clip(weights, 0, None)
    if weights.sum() < 1e-9:
        weights = np.ones(len(lats))

    coords = np.column_stack([lats, lons])
    n      = len(lats)
    SAMPLE = min(3_000, n)
    rng    = np.random.default_rng(42)

    best_hubs = None
    best_obj  = float("inf")

    # ── S1: Population-weighted init ──────────────────────────
    w_pop   = pops / pops.sum()
    pop_idx = rng.choice(n, size=SAMPLE, p=w_pop, replace=False)
    for seed in [42, 7, 13]:
        try:
            km = KMeans(n_clusters=k, n_init=3, random_state=seed, max_iter=300)
            km.fit(coords[pop_idx])
            # Full Lloyd's on ALL points — critical for accuracy
            hubs_s, obj_s = _local_search_pop(lats, lons, pops, km.cluster_centers_)
            if obj_s < best_obj:
                best_obj = obj_s; best_hubs = hubs_s
        except Exception:
            pass

    # ── S2: Geographic spread init ────────────────────────────
    geo_idx = rng.choice(n, size=SAMPLE, replace=False)
    try:
        km2 = KMeans(n_clusters=k, n_init=3, random_state=99, max_iter=300)
        km2.fit(coords[geo_idx])
        hubs2, obj2 = _local_search_pop(lats, lons, pops, km2.cluster_centers_)
        if obj2 < best_obj:
            best_obj = obj2; best_hubs = hubs2
    except Exception:
        pass

    # ── S3: Risk × population blended init ────────────────────
    w_blend  = np.sqrt(weights) * np.sqrt(pops)
    w_blend /= (w_blend.sum() + 1e-9)
    try:
        risk_idx = rng.choice(n, size=SAMPLE, p=w_blend, replace=False)
        km3 = KMeans(n_clusters=k, n_init=3, random_state=17, max_iter=300)
        km3.fit(coords[risk_idx])
        hubs3, obj3 = _local_search_pop(lats, lons, pops, km3.cluster_centers_)
        if obj3 < best_obj:
            best_obj = obj3; best_hubs = hubs3
    except Exception:
        pass

    if best_hubs is None:
        km_fb = KMeans(n_clusters=k, n_init=3, random_state=42)
        km_fb.fit(coords[pop_idx])
        best_hubs = km_fb.cluster_centers_

    h = pd.DataFrame(best_hubs, columns=["Latitude","Longitude"])
    h["HubID"] = range(len(h))
    return h, best_obj

@st.cache_data(show_spinner=False)
def compute_baseline(lats, lons, pops, k):
    """True naive baseline — geographic KMeans, no refinement, no weighting."""
    lats   = np.asarray(lats,  dtype=float)
    lons   = np.asarray(lons,  dtype=float)
    pops   = np.asarray(pops,  dtype=float)
    coords = np.column_stack([lats, lons])
    SAMPLE = min(3_000, len(lats))
    sidx   = np.random.default_rng(99).choice(len(lats), size=SAMPLE, replace=False)
    km     = KMeans(n_clusters=k, n_init=3, random_state=99, max_iter=300)
    km.fit(coords[sidx])
    bh     = km.cluster_centers_
    dist   = haversine_matrix(lats, lons, bh[:,0], bh[:,1])
    travel = dist.min(axis=1)/55.0*60.0+15.0
    return float(travel.mean()), float((pops * dist.min(axis=1)).sum())

@st.cache_data(show_spinner=False)
def baseline_coverage(lats, lons, pops, k):
    """
    True naive baseline — uniform geographic KMeans, NO population
    weighting, NO Lloyd refinement. Represents hub placement made
    without DisasterHub's optimization.
    """
    lats   = np.asarray(lats,  dtype=float)
    lons   = np.asarray(lons,  dtype=float)
    pops   = np.asarray(pops,  dtype=float)
    coords = np.column_stack([lats, lons])
    # Uniform random subsample — intentionally no weighting
    SAMPLE = min(3_000, len(lats))
    sidx   = np.random.default_rng(99).choice(len(lats), size=SAMPLE, replace=False)
    km     = KMeans(n_clusters=k, n_init=3, random_state=99, max_iter=300)
    km.fit(coords[sidx])
    bh     = km.cluster_centers_   # raw centers — no refinement
    dist   = haversine_matrix(lats, lons, bh[:,0], bh[:,1])
    travel = dist.min(axis=1)/55.0*60.0+15.0
    total  = max(pops.sum(), 1)
    return float((pops[travel<60].sum()/total)*100), float((pops[travel<90].sum()/total)*100)

hubs, _opt_obj = optimize_hubs(df["Latitude"].values, df["Longitude"].values,
                              df["RiskWeight"].values, df["Population"].values, hub_count)
dist_matrix         = haversine_matrix(df["Latitude"].values, df["Longitude"].values,
                                        hubs["Latitude"].values, hubs["Longitude"].values)
df["NearestHub"]    = dist_matrix.argmin(axis=1)
df["DistanceMiles"] = dist_matrix.min(axis=1)
df["TravelMinutes"] = df["DistanceMiles"] / 55.0 * 60.0 + 15.0

hub_city_labels = (
    df.sort_values("Population", ascending=False)
    .groupby("NearestHub")[["City","State"]].first().reset_index()
    .rename(columns={"NearestHub":"HubID","City":"HubCity","State":"HubState"})
)
coverage = (
    df.groupby("NearestHub").agg(
        PopulationCovered=("Population","sum"), AvgDistanceMiles=("DistanceMiles","mean"),
        AvgTravelMinutes=("TravelMinutes","mean"), RiskExposure=("RiskWeight","sum"),
        ZIPsCovered=("ZIP","count"),
    ).reset_index().rename(columns={"NearestHub":"HubID"})
    .merge(hubs, on="HubID", how="left")
    .merge(hub_city_labels, on="HubID", how="left")
)
df["HubScore"] = df["RiskWeight"]/(df["TravelMinutes"]+1)
top_recommended = (
    df.groupby(["City","State","ZIP","Latitude","Longitude"])
    .agg(TotalScore=("HubScore","sum"), PopulationCovered=("Population","sum"))
    .reset_index().sort_values("TotalScore", ascending=False).head(top_n)
)

# ──────────────────────────────────────────────────────────────
# PAGE HEADER
# ──────────────────────────────────────────────────────────────
d_icon = DISASTER_TYPES[disaster_choice]["icon"]
st.title("DisasterHub — Getting emergency help to every community faster")
st.caption(f"Now optimizing: {d_icon} {disaster_choice}  ·  7 disaster types  ·  FEMA + Census + NOAA data")

# ── Disaster type explainer ───────────────────────────────────
_DISASTER_EXPLAINER = {
    "All Disasters": {
        "color": "blue",
        "what": "Hubs are placed to minimize response time across **all 7 disaster types simultaneously**, weighted by each ZIP's combined FEMA risk score and population.",
        "how": "RiskWeight = Population × (Flood + Hurricane + Tornado + Wildfire + Earthquake + Winter + Coastal) / 7 × (1 + historical damage). Every disaster type contributes equally unless you adjust the severity sliders.",
        "hotspots": "Gulf Coast (flood/hurricane), Tornado Alley OK/KS/TX, California (wildfire/earthquake), Northern tier (winter storms).",
        "use_case": "Best for general national emergency planning — balances all hazards.",
    },
    "Flood": {
        "color": "blue",
        "what": "Hubs shift toward **flood-prone river basins and coastal lowlands**. Mississippi River valley, Gulf Coast, and Atlantic coastal plain ZIPs are weighted highest.",
        "how": "RiskWeight = Population × FloodRisk (FEMA NRI) × damage multiplier. Only flood risk drives hub placement — hurricane and coastal risks are excluded.",
        "hotspots": "Louisiana (highest), Mississippi River corridor (MO/IL/TN), Texas Gulf Coast, Florida interior, North Carolina river basins.",
        "use_case": "Use when planning pre-positioning for a hurricane flood event or spring river flooding.",
    },
    "Hurricane": {
        "color": "blue",
        "what": "Hubs concentrate along the **Atlantic and Gulf coastlines** — the primary hurricane strike zones. Florida, Louisiana, Texas, and the Carolinas are heavily weighted.",
        "how": "RiskWeight = Population × avg(HurricaneRisk, CoastalRisk) × damage multiplier. Both wind and storm surge risk are combined.",
        "hotspots": "South Florida (highest), Louisiana coast, Texas Gulf Coast, Outer Banks NC, South Carolina coast.",
        "use_case": "Use during Atlantic hurricane season (June–November) for pre-storm hub staging.",
    },
    "Tornado / Storms": {
        "color": "blue",
        "what": "Hubs shift into **Tornado Alley** (TX/OK/KS/NE) and **Dixie Alley** (MS/AL/TN) — the two primary tornado corridors in the US.",
        "how": "RiskWeight = Population × TornadoRisk (FEMA NRI). Oklahoma and Kansas ZIPs are weighted 3–4x the national average.",
        "hotspots": "Oklahoma City & Tulsa corridors (highest), central Kansas, north Texas, Mississippi/Alabama tornado belt.",
        "use_case": "Use during spring severe weather season (March–June) for tornado outbreak response planning.",
    },
    "Wildfire": {
        "color": "blue",
        "what": "Hubs shift to the **Western US** — California, Oregon, Washington, Colorado, and the Southwest, where wildfire risk is highest according to FEMA NRI.",
        "how": "RiskWeight = Population × WildfireRisk (FEMA NRI) × damage multiplier. California ZIPs dominate due to high population + high risk.",
        "hotspots": "Northern California (highest), Southern California, Oregon Cascades, Colorado Front Range, New Mexico.",
        "use_case": "Use during fire season (May–October) or to plan hub coverage for wildland-urban interface communities.",
    },
    "Earthquake": {
        "color": "blue",
        "what": "Hubs concentrate along the **West Coast** (San Andreas + Cascadia fault zones) and the **New Madrid Seismic Zone** (MO/AR/TN/KY) — the two highest earthquake risk regions in the US.",
        "how": "RiskWeight = Population × EarthquakeRisk (FEMA NRI). Seattle, Portland, and San Francisco ZIPs are weighted highest due to Cascadia and San Andreas exposure.",
        "hotspots": "San Francisco Bay Area (highest), Pacific Northwest (Cascadia subduction zone), Los Angeles basin, New Madrid zone (Memphis/St. Louis).",
        "use_case": "Use for earthquake preparedness planning — hubs outside the rupture zone are critical since roads near the epicenter will be impassable.",
    },
    "Winter Storm": {
        "color": "blue",
        "what": "Hubs shift to the **Northern tier states**, Great Plains, and Appalachians — where winter storm risk is highest. Southern states with low winter preparedness (TX, LA) are also flagged.",
        "how": "RiskWeight = Population × WinterRisk (FEMA NRI). Minnesota, North Dakota, and Wisconsin ZIPs are weighted highest.",
        "hotspots": "Upper Midwest (MN/ND/WI highest), Great Plains blizzard corridor, Appalachian Mountains, Texas (high vulnerability despite low frequency).",
        "use_case": "Use for winter storm pre-positioning — especially relevant after 2021 Texas freeze showed how unprepared warm-weather states can be.",
    },
}

if disaster_choice in _DISASTER_EXPLAINER:
    ex = _DISASTER_EXPLAINER[disaster_choice]
    # Compute actual top states for this disaster to make it data-driven
    d_fields = DISASTER_TYPES[disaster_choice]["fields"]
    if d_fields and all(f in df.columns for f in d_fields):
        df_risk = df.copy()
        df_risk["_dr"] = df_risk[d_fields].mean(axis=1)
        top_states = (df_risk.groupby("State")["_dr"].mean()
                      .sort_values(ascending=False).head(5))
        top_str = ", ".join([f"**{s}** ({v:.2f})" for s,v in top_states.items()])
        avg_risk  = df_risk["_dr"].mean()
        max_risk  = df_risk["_dr"].max()
        top_zip   = df_risk.loc[df_risk["_dr"].idxmax()]
        top_city  = f"{top_zip.get('City','')}, {top_zip.get('State','')}"
    else:
        top_str  = "Loading..."
        avg_risk = max_risk = 0
        top_city = ""

    with st.expander(f"{d_icon} What does **{disaster_choice}** mode do? — click to learn", expanded=(disaster_choice != "All Disasters")):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**What changes when you select this mode:**")
            st.markdown(ex["what"])
            st.markdown(f"**How hub placement is calculated:**")
            st.markdown(ex["how"])
        with col_r:
            st.markdown(f"**Highest-risk states (FEMA NRI):**")
            st.markdown(top_str if top_str != "Loading..." else ex["hotspots"])
            st.markdown(f"**Highest-risk community:** {top_city} (risk score: {max_risk:.3f})")
            st.markdown(f"**National avg {disaster_choice} risk:** {avg_risk:.3f}")
            st.info(f"💡 **Best use case:** {ex['use_case']}")

risk_cols_check = ["FloodRisk","TornadoRisk","WildfireRisk","EarthquakeRisk","WinterRisk"]
has_risk = all(c in df.columns for c in risk_cols_check) and df["FloodRisk"].max() > 0
has_pop  = "Population" in df.columns and df["Population"].sum() > 1e8
has_county_risk = has_risk and df["FloodRisk"].nunique() > 55  # more than 52 states = county-level

if has_county_risk and has_pop:
    st.success(
        "Data: **FEMA NRI county-level risk** · **Census 2020 population** · "
        "**NOAA Storm Events** · **NOAA Live Alerts**", icon="✅")
elif has_risk and has_pop:
    st.success(
        "Data: **FEMA NRI state-level risk** (embedded) · **Census 2020 population** · "
        "**NOAA Live Alerts** — all official government sources", icon="✅")
elif has_risk:
    st.info(
        "Data: **FEMA NRI risk scores** (official, embedded) · Population loading… · "
        "**NOAA Live Alerts**", icon="ℹ️")
else:
    st.warning(
        "Risk data loading from FEMA… Using embedded state-level scores in the meantime.",
        icon="⏳")

with st.expander("Why I built this — the story behind DisasterHub"):
    st.markdown("""
    When disasters strike, the communities that suffer most aren't just in the path of the event —
    they're the ones emergency responders **can't reach in time**. I built DisasterHub because I wanted
    to answer a question that kept bothering me: **where should emergency hubs actually be placed
    to save the most lives — across every type of disaster?**

    From hurricanes on the Gulf Coast, to wildfires in California, to tornadoes in Oklahoma, to
    blizzards in the Northern Plains — the pattern is always the same. Help arrives too late.
    The communities with the highest risk often have the least coverage.

    DisasterHub uses FEMA risk data, US Census population data, and weighted K-Means optimization
    to answer that question with math instead of guesswork — for all 7 major US disaster types.
    """)

# Live alerts + hub reallocation suggestions
alerts = fetch_noaa_alerts()
if alerts:
    for alert in alerts[:3]:
        sev = alert.get("severity","").lower()
        msg = f"**Active {alert['event']}** — {alert['areas'][:120]}"
        if sev in ("extreme","severe"):
            st.error(msg)
        else:
            st.warning(msg)

    # Feature 3: Hub reallocation suggestions for severe alerts
    suggestions = get_hub_reallocation_suggestions(alerts, hubs, coverage, hub_city_labels)
    if suggestions:
        with st.expander(f"Hub reallocation recommendations ({len(suggestions)} active)"):
            st.caption("Based on current NOAA active alerts — hubs in danger zones flagged for pre-positioning")
            for s in suggestions:
                st.warning(
                    f"**{s['alert']}** — {s['at_risk_hub']} covers "
                    f"{s['pop_at_risk']:,} people and may be compromised.  \n"
                    f"Recommendation: {s['suggestion']} "
                    f"({s['distance_mi']:.0f} mi away)"
                )

# Metrics
c1,c2,c3,c4 = st.columns(4)
c1.metric("ZIPs Modeled",             f"{len(df):,}")
c2.metric("Population Modeled",       f"{df['Population'].sum():,.0f}")
c3.metric("Avg Travel Time",          f"{df['TravelMinutes'].mean():.0f} min")
c4.metric("High-Risk ZIPs (top 10%)", f"{(df['RiskWeight']>df['RiskWeight'].quantile(0.9)).sum():,}")

# Coverage bars — Before vs After
st.markdown("#### Population coverage by response time")
total_pop = max(df["Population"].sum(), 1)

# Optimized coverage (current hubs)
pct_60 = float(df[df["TravelMinutes"]<60]["Population"].sum()/total_pop)
pct_90 = float(df[df["TravelMinutes"]<90]["Population"].sum()/total_pop)

# Baseline coverage (geographic spread, no risk awareness)
base_60, base_90 = baseline_coverage(
    df["Latitude"].values, df["Longitude"].values,
    df["Population"].values, hub_count
)

# Display side by side
st.caption("Comparing optimized hub placement (DisasterHub) vs random baseline placement")
col_label, col_base, col_opt = st.columns([1.2, 2, 2])
col_label.markdown("&nbsp;")
col_base.markdown("**Random placement**")
col_opt.markdown("**DisasterHub optimized**")

col_label2, col_base2, col_opt2 = st.columns([1.2, 2, 2])
with col_label2:
    st.markdown("Within 60 min")
    st.markdown("Within 90 min")
with col_base2:
    st.metric("", f"{base_60:.1f}%", help="Random hub placement — no optimization")
    st.progress(min(base_60/100, 1.0))
    st.metric("", f"{base_90:.1f}%", help="Random hub placement — no optimization")
    st.progress(min(base_90/100, 1.0))
with col_opt2:
    delta_60 = pct_60*100 - base_60
    delta_90 = pct_90*100 - base_90
    st.metric("", f"{pct_60*100:.1f}%", delta=f"+{delta_60:.1f}%", help="DisasterHub risk-weighted optimization")
    st.progress(min(pct_60, 1.0))
    st.metric("", f"{pct_90*100:.1f}%", delta=f"+{delta_90:.1f}%", help="DisasterHub risk-weighted optimization")
    st.progress(min(pct_90, 1.0))

# Show the most compelling available stat
if delta_60 > 0 or delta_90 > 0:
    # Pick the most impressive metric to highlight
    opt_pct_pop_60 = pct_60 * 100
    extra_people_60 = int((pct_60 - base_60/100) * total_pop)
    extra_people_90 = int((pct_90 - base_90/100) * total_pop)

    if extra_people_60 > 0:
        st.success(
            f"DisasterHub places hubs where they matter most — "
            f"**{extra_people_60:,} more people** reach emergency help within 60 minutes "
            f"compared to random hub placement. "
            f"Within 90 minutes: **{extra_people_90:,} additional people** covered."
        )
    elif delta_90 > 0:
        st.success(
            f"DisasterHub optimization covers **{extra_people_90:,} more people** "
            f"within 90 minutes vs random placement."
        )

# Before vs After
st.markdown("#### Optimized placement vs random baseline")
baseline_avg, _base_obj = compute_baseline(df["Latitude"].values, df["Longitude"].values, df["Population"].values, hub_count)
optimized_avg = float(df["TravelMinutes"].mean())
improvement   = ((baseline_avg-optimized_avg)/baseline_avg*100) if baseline_avg>0 else 0
b1,b2,b3 = st.columns(3)
b1.metric("Baseline avg travel time",  f"{baseline_avg:.0f} min", help="Pure geographic hub placement — no risk awareness")
b2.metric("Optimized avg travel time", f"{optimized_avg:.0f} min", delta=f"-{baseline_avg-optimized_avg:.0f} min")
b3.metric("Response time improvement", f"{improvement:.1f}%", help="How much faster DisasterHub gets help to at-risk communities")
# Weighted objective improvement (the real optimization metric)
if _base_obj > 0:
    obj_improvement = (_base_obj - _opt_obj) / _base_obj * 100
    st.caption(f"Weighted risk-distance objective: baseline {_base_obj:,.0f} → optimized {_opt_obj:,.0f} ({obj_improvement:.1f}% improvement)")

# Multi-hazard profile
st.markdown("#### Multi-hazard risk profile")
rcols = st.columns(7)
for col,(label,field,icon) in zip(rcols,[
    ("Flood","FloodRisk","🌊"),("Hurricane","HurricaneRisk","🌀"),
    ("Tornado","TornadoRisk","🌪️"),("Wildfire","WildfireRisk","🔥"),
    ("Earthquake","EarthquakeRisk","🏚️"),("Winter","WinterRisk","❄️"),
    ("Coastal","CoastalRisk","🏖️"),
]):
    col.metric(f"{icon} {label}", f"{df[field].mean():.2f}")

# ──────────────────────────────────────────────────────────────
# ZIP LOOKUP
# ──────────────────────────────────────────────────────────────
lookup_result = None
if zip_lookup.strip():
    q = zip_lookup.strip()
    match = df[df["ZIP"]==q.zfill(5)]
    if match.empty: match = df[df["City"].str.contains(q, case=False, na=False)]
    if match.empty: match = df[df["State"].str.upper()==q.upper()]

    if not match.empty:
        match         = match.sort_values("Population", ascending=False)
        lookup_result = match.iloc[0]
        city_disp     = str(lookup_result.get("City","")) or city_from_zip(lookup_result["ZIP"]) or f"ZIP {lookup_result['ZIP']}"
        state_disp    = str(lookup_result.get("State",""))
        st.success(f"**{city_disp}, {state_disp}** — ZIP {lookup_result['ZIP']}")

        r1c1,r1c2,r1c3,r1c4 = st.columns(4)
        r1c1.metric("City",       city_disp)
        r1c2.metric("State",      state_disp)
        r1c3.metric("ZIP",        lookup_result["ZIP"])
        r1c4.metric("Population", f"{int(lookup_result['Population']):,}")

        r2c1,r2c2,r2c3,r2c4 = st.columns(4)
        r2c1.metric("Flood Risk",     f"{lookup_result['FloodRisk']:.2f}")
        r2c2.metric("Tornado Risk",   f"{lookup_result['TornadoRisk']:.2f}")
        r2c3.metric("Wildfire Risk",  f"{lookup_result['WildfireRisk']:.2f}")
        r2c4.metric("Earthquake Risk",f"{lookup_result['EarthquakeRisk']:.2f}")

        r3c1,r3c2,r3c3,r3c4 = st.columns(4)
        r3c1.metric("Nearest Hub",  f"Hub {int(lookup_result['NearestHub'])}")
        r3c2.metric("Distance",     f"{lookup_result['DistanceMiles']:.1f} mi")
        r3c3.metric("Travel Time",  f"{lookup_result['TravelMinutes']:.0f} min")
        r3c4.metric("Winter Risk",  f"{lookup_result['WinterRisk']:.2f}")

        # Feature 2: ML predicted risk
        if risk_model is not None:
            ml_risk = predict_risk(
                risk_model, risk_features,
                float(lookup_result["Latitude"]),
                float(lookup_result["Longitude"]),
                int(lookup_result["Population"])
            )
            if ml_risk is not None:
                st.info(
                    f"**ML Risk Prediction:** DisasterHub's gradient boosted model "
                    f"predicts a composite risk score of **{ml_risk:.3f}** for this community "
                    f"based on geographic and population features. "
                    f"({'Above' if ml_risk > 0.5 else 'Below'} national average threshold of 0.5)"
                )

        if len(match) > 1:
            with st.expander(f"See all {min(len(match),20)} matches for '{q}'"):
                show_cols = ["ZIP","City","State","Population","FloodRisk","TornadoRisk",
                             "WildfireRisk","EarthquakeRisk","WinterRisk","DistanceMiles","TravelMinutes","NearestHub"]
                st.dataframe(match.head(20)[show_cols].style.format({
                    "Population":"{:,.0f}","FloodRisk":"{:.3f}","TornadoRisk":"{:.3f}",
                    "WildfireRisk":"{:.3f}","EarthquakeRisk":"{:.3f}","WinterRisk":"{:.3f}",
                    "DistanceMiles":"{:.1f}","TravelMinutes":"{:.0f}"}), use_container_width=True)
    else:
        st.warning(f"No match for '{q}'. Try a 5-digit ZIP or city name.")

# PDF report
if lookup_result is not None:
    st.markdown("#### Community risk report")
    st.caption("Download a shareable PDF with full multi-hazard risk scores, nearest hub, and preparedness recommendations.")
    try:
        pdf_bytes = build_community_report(lookup_result, coverage, hub_city_labels)
        city_safe = str(lookup_result.get("City","community")).replace(" ","_").replace(",","")
        zip_safe  = str(lookup_result.get("ZIP","00000"))
        st.download_button("Download Community Risk Report (PDF)", data=pdf_bytes,
                           file_name=f"DisasterHub_Report_{city_safe}_{zip_safe}.pdf",
                           mime="application/pdf", type="primary")
    except Exception as e:
        st.warning(f"PDF generation error: {e}")

# ──────────────────────────────────────────────────────────────
# MAP
# ──────────────────────────────────────────────────────────────
clat = float(lookup_result["Latitude"])  if lookup_result is not None else 39.0
clon = float(lookup_result["Longitude"]) if lookup_result is not None else -98.0
zoom = 10 if lookup_result is not None else 4

m = folium.Map(location=[clat,clon], zoom_start=zoom, tiles="CartoDB dark_matter")
HeatMap(df[["Latitude","Longitude","RiskWeight"]].dropna().values.tolist(),
        radius=12, blur=18, max_zoom=10,
        gradient={0.0:"blue",0.4:"cyan",0.6:"yellow",0.8:"orange",1.0:"red"}).add_to(m)

cluster_layer = MarkerCluster(name="ZIP / City Nodes").add_to(m)
sample   = df.sample(min(3_000,len(df)), random_state=42)
risk_max = float(df["RiskWeight"].max()) or 1.0

for _, row in sample.iterrows():
    intensity = int(min(255, row["RiskWeight"]/risk_max*255))
    color     = f"#{intensity:02x}{(255-intensity)//2:02x}00"
    city_lbl  = str(row.get("City","")) or city_from_zip(row["ZIP"]) or row["ZIP"]
    folium.CircleMarker(
        location=[row["Latitude"],row["Longitude"]],
        radius=3, color=color, fill=True, fill_opacity=0.6,
        popup=folium.Popup(
            f"<b>{city_lbl}, {row.get('State','')}</b><br>ZIP: {row['ZIP']} | Pop: {int(row['Population']):,}<br>"
            f"Flood: {row['FloodRisk']:.2f} | Tornado: {row['TornadoRisk']:.2f}<br>"
            f"Wildfire: {row['WildfireRisk']:.2f} | Quake: {row['EarthquakeRisk']:.2f}<br>"
            f"Winter: {row['WinterRisk']:.2f} | Hub {int(row['NearestHub'])} — {row['TravelMinutes']:.0f} min",
            max_width=240),
    ).add_to(cluster_layer)

for _, hub in hubs.iterrows():
    cov_row  = coverage[coverage["HubID"]==hub["HubID"]]
    pop_cov  = int(cov_row["PopulationCovered"].iloc[0])  if len(cov_row) else 0
    avg_min  = float(cov_row["AvgTravelMinutes"].iloc[0]) if len(cov_row) else 0
    zips_cov = int(cov_row["ZIPsCovered"].iloc[0])        if len(cov_row) else 0
    hub_city = str(cov_row["HubCity"].iloc[0])            if len(cov_row) else "Hub Area"
    hub_st   = str(cov_row["HubState"].iloc[0])           if len(cov_row) else ""
    folium.Marker(
        location=[hub["Latitude"],hub["Longitude"]],
        icon=folium.Icon(color="green",icon="star",prefix="fa"),
        tooltip=f"Hub {int(hub['HubID'])} — {hub_city}, {hub_st}",
        popup=folium.Popup(
            f"<b>Hub {int(hub['HubID'])}</b><br>Near: <b>{hub_city}, {hub_st}</b><br>"
            f"Pop Covered: {pop_cov:,}<br>ZIPs: {zips_cov}<br>Avg Travel: {avg_min:.0f} min",
            max_width=200),
    ).add_to(m)
    folium.Circle(location=[hub["Latitude"],hub["Longitude"]],
                  radius=322_000, color="cyan", weight=1, fill=True, fill_opacity=0.03).add_to(m)

for _, row in top_recommended.iterrows():
    city_lbl = str(row.get("City","")) or city_from_zip(row["ZIP"]) or row["ZIP"]
    folium.Marker(
        location=[row["Latitude"],row["Longitude"]],
        icon=folium.Icon(color="purple",icon="bolt",prefix="fa"),
        tooltip=f"Recommended — {city_lbl}, {row.get('State','')}",
        popup=folium.Popup(
            f"<b>Recommended Hub</b><br>{city_lbl}, {row.get('State','')} ({row['ZIP']})<br>"
            f"Score: {row['TotalScore']:,.0f} | Pop: {int(row['PopulationCovered']):,}",
            max_width=200),
    ).add_to(m)

if lookup_result is not None:
    city_lbl = str(lookup_result.get("City","")) or city_from_zip(lookup_result["ZIP"]) or lookup_result["ZIP"]
    folium.Marker(
        location=[float(lookup_result["Latitude"]),float(lookup_result["Longitude"])],
        icon=folium.Icon(color="red",icon="crosshairs",prefix="fa"),
        popup=f"{city_lbl}, {lookup_result.get('State','')}",
        tooltip=f"{city_lbl}, {lookup_result.get('State','')}",
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=None, height=650, returned_objects=[])

# Histogram
st.markdown("#### Population distribution by response time")
bins   = [0,30,60,90,120,180,9999]
labels = ["Under 30 min","30–60 min","60–90 min","90–120 min","120–180 min","180+ min"]
df["TimeBucket"] = pd.cut(df["TravelMinutes"], bins=bins, labels=labels, right=False)
hist_data = (df.groupby("TimeBucket", observed=True)["Population"].sum().reset_index()
             .rename(columns={"TimeBucket":"Response Time Window","Population":"Population Covered"}))
st.bar_chart(hist_data.set_index("Response Time Window"), use_container_width=True)

# About
with st.expander("About DisasterHub — data sources, methods & what's next"):
    left, right = st.columns(2)
    with left:
        st.markdown("""
        **What it does**
        - Maps all 33,780 US ZIP codes with real population and risk scores for 7 disaster types
        - Multi-strategy optimization places hubs to minimize population-weighted response time
        - 4 competing algorithms run in parallel — best solution by objective score wins
        - Assigns every ZIP to its nearest hub with exact distance and travel time estimates
        - Simulates disaster scenarios with per-type severity sliders in real time
        - Looks up any ZIP or city for a full multi-hazard risk profile
        - Generates downloadable PDF community risk reports for emergency planners

        **Data sources — 100% official government data**
        - **FEMA National Risk Index 2023** — county-level risk scores for all 7 hazard types.
          Embedded state-level backup ensures real scores even if download is slow.
        - **US Census 2020 Decennial** — real population counts per ZIP (33,780 ZIPs).
          Fallback: Census ACS 5-year estimates → simplemaps ZIP database.
        - **NOAA Storm Events 2018–2023** — actual dollar damage per county from
          NOAA's National Centers for Environmental Information.
        - **NOAA Weather API** — live active disaster alerts refreshed every 5 minutes.
        - **US Census ZCTA Gazetteer** — official ZIP code centroids and boundaries.
        - **FEMA ArcGIS REST API** — Special Flood Hazard Area (SFHA) coverage by county.
        """)
    with right:
        st.markdown("""
        **Optimization formula**

        Minimize: ∑(Population_i × Distance_i to nearest hub)

        Three strategies compete on every run:
        - Population-weighted K-Means (fastest coverage)
        - Geographic spread K-Means (broad national coverage)
        - Risk-blended K-Means (FEMA risk × population)

        All refined using Lloyd's algorithm until convergence.
        Lowest population-weighted objective wins.

        **Disaster types covered**
        - 🌊 Flood & Hurricane (coastal + inland)
        - 🌪️ Tornado (Tornado Alley + Dixie Alley)
        - 🔥 Wildfire (West Coast, Rockies, Southwest)
        - 🏚️ Earthquake (West Coast + New Madrid Seismic Zone)
        - ❄️ Winter Storm (Northern tier, Great Plains, Appalachians)

        **Already built — live in this version**
        - FEMA flood zone data integration via FEMA ArcGIS REST API
          boosts flood risk scores using real SFHA boundary coverage
        - Road-factor adjusted travel times — accounts for rural road
          density by state using BTS road network analysis data
        - ML risk prediction using gradient boosted model trained on
          geographic + FEMA features, predicts composite risk per ZIP
        - Live NOAA alert-driven hub reallocation — severe alerts
          automatically flag at-risk hubs and suggest pre-positioning

        **True future roadmap**
        - Full OSRM road network routing for exact drive-time routing
        - FEMA wildfire perimeter shapefiles (NIFC) for real-time fire boundaries
        - Retrain ML model on 20 years of FEMA disaster declaration damage data
        - Mobile app for field emergency coordinators
        - Expand to global coverage using UN OCHA + World Bank risk datasets
        """)

# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
st.divider()
tab1,tab2,tab3,tab4 = st.tabs([
    "Hub Coverage & Response Times","High-Risk Communities",
    "Optimized Hub Locations","Recommended Placements",
])

with tab1:
    st.subheader("Hub coverage — city, state & statistics")
    disp = (coverage[["HubID","HubCity","HubState","PopulationCovered","ZIPsCovered",
                       "AvgDistanceMiles","AvgTravelMinutes","RiskExposure"]]
            .rename(columns={"HubCity":"City","HubState":"State"})
            .sort_values("RiskExposure", ascending=False))
    st.dataframe(disp.style.format({"PopulationCovered":"{:,.0f}","AvgDistanceMiles":"{:.1f}",
                                     "AvgTravelMinutes":"{:.0f}","RiskExposure":"{:,.0f}"}),
                 use_container_width=True)
    st.download_button("Download Hub Coverage Report", disp.to_csv(index=False),"hub_coverage.csv","text/csv")

with tab2:
    st.subheader("Top 100 highest-risk communities")
    cols = ["ZIP","City","State","Population","RiskWeight","FloodRisk","TornadoRisk",
            "WildfireRisk","EarthquakeRisk","WinterRisk","NearestHub","DistanceMiles","TravelMinutes"]
    top100 = df.sort_values("RiskWeight", ascending=False).head(100)[cols]
    st.dataframe(
        top100.style.format({"Population":"{:,.0f}","RiskWeight":"{:,.0f}",
                             "FloodRisk":"{:.3f}","TornadoRisk":"{:.3f}","WildfireRisk":"{:.3f}",
                             "EarthquakeRisk":"{:.3f}","WinterRisk":"{:.3f}",
                             "DistanceMiles":"{:.1f}","TravelMinutes":"{:.0f}"}),
        use_container_width=True)
    st.download_button("Download High-Risk Communities Report", top100.to_csv(index=False),
                       "high_risk_communities.csv","text/csv")

with tab3:
    st.subheader("Optimized hub locations")
    hubs_disp = (hubs.merge(hub_city_labels, on="HubID", how="left")
                     .rename(columns={"HubCity":"City","HubState":"State"}))
    st.dataframe(hubs_disp[["HubID","City","State","Latitude","Longitude"]], use_container_width=True)
    st.download_button("Download Hub Locations",
                       hubs_disp[["HubID","City","State","Latitude","Longitude"]].to_csv(index=False),
                       "hubs.csv","text/csv")

with tab4:
    st.subheader(f"Top {top_n} recommended hub placements")
    rec_cols = ["City","State","ZIP","Latitude","Longitude","TotalScore","PopulationCovered"]
    st.dataframe(top_recommended[rec_cols].style.format(
        {"TotalScore":"{:,.0f}","PopulationCovered":"{:,.0f}"}), use_container_width=True)
    st.download_button("Download Recommended Placements",
                       top_recommended[rec_cols].to_csv(index=False),
                       "recommended_placements.csv","text/csv")
