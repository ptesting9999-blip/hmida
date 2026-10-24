"""Free intent classification without any AI model: phrase rules over the post text.

Less nuanced than Claude or a local model, but costs nothing and runs anywhere. It produces
the same IntentAssessment, so the rest of the pipeline is unchanged.

The rules work sentence by sentence and lean towards precision: a post that only looks like a
request ("Need a social media manager? I have 2 spots open") is rejected, because every
accepted post costs crawling and, in the cloud setup, a paid verification credit.
"""

from __future__ import annotations

import re
from typing import Optional

from leadgen.config import Config
from leadgen.enrich.domain import name_matches, normalize_domain
from leadgen.intent import SELLER_PATTERNS, IntentAssessment, ServiceKind
from leadgen.models import RawPost

# -- vocabulary -------------------------------------------------------------------------------

_SERVICE = (
    r"social\s*media(?:\s*(?:managers?|management|marketing|marketers?|agenc(?:y|ies)|person|help|experts?|specialists?|"
    r"presence|accounts?|content|strategy|strategist))?|socials|smm|"
    r"(?:facebook|fb|meta|instagram|ig|tiktok|google|linkedin|youtube|paid|pinterest|snapchat)\s*ads?"
    r"(?:\s*(?:managers?|specialists?|experts?|agenc(?:y|ies)|person|help|campaigns?|buyers?))?|"
    r"ads?\s*managers?|ads?\s*specialists?|media\s*buyers?|ppc(?:\s*agenc(?:y|ies))?|"
    r"(?:digital\s*)?marketing\s*(?:agenc(?:y|ies)|help|person|expert|freelancer|consultant)|digital\s*marketers?|"
    r"content\s*creators?|ugc\s*creators?|instagram\s*(?:marketing|growth|managers?)|growth\s*marketers?"
)
# Things an owner asks someone to run for them: "someone to manage our socials".
_CHANNEL = r"(?:instagram|insta|ig|tiktok|socials|social media|social|facebook|fb|ads|meta ads|google ads|marketing|content|pages?|accounts?)"
_ASK = (
    r"looking\s+for|looking\s+to\s+(?:hire|outsource|find)|need(?:s|ed|ing)?|hiring|seeking|searching\s+for|in\s+search\s+of|"
    r"want\s+to\s+(?:hire|outsource)|outsourc(?:e|ing)|(?:can\s+)?any(?:one|body)\s+(?:recommend|know|suggest)|"
    r"recommend(?:ations?)?\s+(?:for|on)|who\s+(?:can|could)\s+(?:help|run|manage)|who\s+do(?:es)?\s+(?:you|your)|"
    r"any\s+(?:recommendations|suggestions|good|reliable)|where\s+(?:can|do)\s+i\s+find|"
    r"(?:would|could)\s+(?:love|use)\s+(?:some\s+)?help\s+with|need\s+a\s+replacement"
)
# Asking for names is a request even as a question ("Can anyone recommend a TikTok ads agency?").
_RECOMMEND = re.compile(
    r"\b(?:recommend|suggest|know (?:a|any|of)|who do(?:es)? (?:you|your)|any (?:good|reliable)|where (?:can|do) i find|"
    r"who (?:can|could) (?:help|run|manage)|any (?:recommendations|suggestions)|referrals?)\b",
    re.I,
)
_REQUEST_RE = re.compile(rf"\b(?:{_ASK})\b(?:\W+\w+){{0,8}}?\W+(?:an?\s+|some\s+|our\s+|my\s+|the\s+)?(?:{_SERVICE})\b", re.I)
_WANTED_RE = re.compile(rf"\b(?:{_SERVICE})\s+(?:wanted|needed|required)\b", re.I)  # classified-ad style
_SOMEONE_RE = re.compile(
    rf"\b(?:someone|somebody|a person|an expert|a freelancer|help)\s+to\s+(?:run|manage|handle|do|create|take\s+over|grow|post)"
    rf"\s+(?:\w+\s+){{0,2}}?(?:my|our|the)\s+(?:\w+\s+)?{_CHANNEL}\b",
    re.I,
)
_SERVICE_RE = re.compile(rf"\b(?:{_SERVICE})\b", re.I)
# "Any social media agencies you'd recommend for a med spa?"
_RECOMMEND_AFTER_RE = re.compile(
    rf"\b(?:{_SERVICE})\b[^.?!\n]{{0,40}}?\b(?:you'?d|you would|would you|can you|could you|do you|anyone|y'?all) recommend\b", re.I
)
# "Just fired our Facebook ads agency. Need a replacement ASAP."
_SWITCH_RE = re.compile(
    rf"\b(?:fired|left|dropped|let go of|leaving|parted ways with|done with|unhappy with|switching from)\s+(?:our|my)\s+"
    rf"(?:\w+\s+){{0,3}}?(?:agency|manager|marketer|freelancer|media buyer|{_SERVICE})\b[^\n]{{0,80}}?"
    rf"\b(?:need|looking for|want|recommend)\w*\s+(?:a\s+)?(?:new|replacement|another|better)\b",
    re.I,
)
# Learners and tool shoppers: "a good course on Facebook ads", "a social media management tool".
_NOT_A_SERVICE_RE = re.compile(
    r"\b(?:course|courses|class|classes|tutorials?|books?|tools?|software|apps?|templates?|alternative|plugins?|extensions?|"
    r"scheduler|scheduling|tips|advice|resources?|ebooks?|certifications?|bootcamp|mentor(?:ship)?|internships?)\b|"
    r"\bstart(?:ing)? my own (?:smma|agency|business)\b|\bfor beginners\b",
    re.I,
)
_ASK_WORD = r"(?:need|needs|needed|looking|hiring|seeking|want|wanted|recommend|outsource)"

# -- sellers, job seekers, content --------------------------------------------------------------

_MARKETING = r"(?:marketing|social media|smm|ads|advertising|digital|media|growth|content|creative|branding|ppc|seo|ugc)"
_SELLER_STRONG = re.compile(
    r"\b(?:i'?m|i am|we are|we're)\s+an?\s+(?:freelance\s+|certified\s+|experienced\s+|full[- ]service\s+)?"
    r"(?:social media|smm|facebook ads?|meta ads?|google ads?|tiktok ads?|paid ads?|ads|ppc|digital marketing|marketing|content|ugc)\s+"
    r"(?:manager|marketer|agency|specialist|expert|consultant|strategist|creator|freelancer|buyer|pro)\b|"
    rf"\b(?:my|our)\s+(?:new\s+)?(?:\w+\s+){{0,2}}?{_MARKETING}\s+(?:agency|company|firm|studio|business|services)\b|"
    r"\bour agency\b|\b(?:my|our) (?:portfolio|rates|packages)\b|\bavailable for (?:work|hire|projects)\b|\bfor hire\b|"
    r"\b(?:taking|accepting) (?:on )?(?:new )?clients\b|\b\d+\s+(?:spots?|slots?)\s+(?:open|left|available)\b|"
    r"\bspots? (?:open|left|available)\b|\blink in (?:my )?bio\b|\bbook (?:a|your) (?:free )?(?:call|consultation|strategy)\b|"
    r"\bfree (?:strategy (?:session|call)|audit)\b|\brequest a proposal\b|\bget (?:a|your) (?:free )?(?:quote|proposal)\b|"
    r"\bour team is here\b|\bwe deliver\b|\btailored to your\b|\b(?:grow|scale|boost) your (?:business|brand|sales|revenue)\b|"
    r"\bhelps (?:businesses|brands|companies|owners|clients)\b|\byou'?re in the right place\b|\bhelp your (?:business|brand)\b|"
    r"\bthat can help your\b|\bif your (?:business|brand|company) is looking\b|"
    r"\b(?:across|for|manage) (?:our|multiple) client(?:s'|'s)? (?:accounts|campaigns|brands)\b|\bour clients'? \w+|"
    r"\bacross (?:selected |our |multiple |several )?(?:brands|clients|client accounts)\b|\bmedia buying team\b|\b(?:social media|marketing|advertising|digital|creative|ads) (?:company|agency|firm) "
    r"(?:is |are )?(?:looking|hiring|seeking)\b",
    re.I,
)
# Claims of doing marketing for others. The service must come before any request word, so
# "I help coaches find balance and I need a social media manager" stays a buyer.
_TEMPERED = rf"(?:(?!\b{_ASK_WORD}\b)[^.?!\n]){{0,80}}?"
_SELLER_CLAIM = re.compile(
    rf"\b(?:i|we)\s+(?:can\s+)?(?:help|manage|create|make|grow|run|handle|scale|specialize|specialise)\b{_TEMPERED}"
    rf"\b(?:for|with)\s+(?:[\w&,]+\s+){{0,3}}?(?:businesses|brands|clients|companies|coaches|creators|owners|gyms|salons|"
    rf"restaurants|clinics|startups)\b|"
    rf"\b(?:we offer|(?:my|our) services|services include|helping (?:businesses|brands|owners|salons|gyms|restaurants))\b"
    rf"{_TEMPERED}\b(?:{_SERVICE}|marketing|followers|instagram|tiktok|ads)\b|"
    rf"\b(?:i|we)\s+(?:can\s+)?help\s+(?:\w+\s+){{0,3}}?(?:businesses|brands|owners|companies|coaches|clinics|gyms|salons|"
    rf"restaurants){_TEMPERED}\b(?:{_SERVICE}|marketing|followers|instagram|tiktok|ads|customers|clients|sales|leads)\b|"
    rf"\b(?:do|does|are)\s+(?:you|your\s+(?:business|brand|company|store|gym|salon|restaurant))\s+(?:need|looking|want|struggling)\b|"
    rf"\byour\s+(?:business|brand|company|store|shop|gym|salon|restaurant)\s+(?:\w+\s+){{0,2}}?needs?\b",
    re.I,
)
# People looking for a marketing job or for clients, not for help.
_JOB_SEEKER_RE = re.compile(
    r"\blooking for (?:a |an |some )?(?:new )?(?:work|job|jobs|employment|opportunit\w*|clients|gigs?|internships?)\b|"
    r"\bmy client(?:,| is| has)\b|\bon behalf of (?:my|our|a) client\b|\bfortune 500\b|\bi'?m a recruiter\b|"
    r"\b(?:amazon|nike|apple|microsoft|netflix|walmart|adidas|coca-cola|pepsico?|disney|uber|airbnb|spotify|tesla|samsung|"
    r"sony|ibm|oracle|salesforce|adobe|accenture|unilever|nestl[eé]|l'or[eé]al|starbucks|mcdonald'?s|google|meta platforms)"
    r"\s+(?:is|are)\s+(?:now\s+)?(?:hiring|looking)\b|"
    r"\b(?:any(?:one|body)|who(?:'s| is)|is any(?:one|body)) (?:here )?hiring\b|\bneed (?:a )?job\b|\bmy (?:resume|cv)\b|"
    r"\bopen to (?:work|opportunities|new roles)\b|#opentowork\b|"
    rf"\b(?:work|job|career|role|position|internship) as an?\s+(?:freelance\s+|remote\s+)?(?:{_SERVICE})\b|"
    rf"\b(?:seeking|looking for)\s+(?:a\s+|an\s+)?(?:remote\s+|part[- ]time\s+|full[- ]time\s+|entry[- ]level\s+|freelance\s+)?"
    rf"(?:{_SERVICE})\s+(?:positions?|roles?|jobs?|internships?|gigs?|work|opportunit\w*)\b",
    re.I,
)
# The poster is the one hiring: "[Hiring]", "we're hiring", "at our bakery", a pay rate.
_HIRING_CUE = re.compile(
    r"\[\s*hiring\s*\]|\b(?:we|i)(?:'re|'m| are| am)?\s+(?:now\s+)?hiring\b|\bhiring\s*:|\bjoin (?:our|my) team\b|"
    r"\bat (?:our|my) (?:\w+\s+)?(?:bakery|salon|spa|gym|studio|cafe|restaurant|shop|store|clinic|practice|business|company|boutique)\b|"
    r"[$£€]\s?\d+(?:\.\d+)?\s*(?:/|per\s+)\s*(?:hr|hour)\b",
    re.I,
)
# Content that only uses a request phrase as a hook: "Hiring a social media manager? Here's what to check."
_CONTENT_RE = re.compile(
    r"\b\d+\s+(?:ways|reasons|signs|things|questions|tips|mistakes|steps|secrets|lessons)\b|\bhow to (?:hire|choose|find|pick|grow|get|"
    r"market|run|scale|boost|increase|reduce)\b|\btips (?:for|to|on) (?:hiring|choosing|finding|growing|picking)\b|"
    r"\bwhat to look for\b|\bfactors? to consider\b|\bbefore you hire\b|\bthings to (?:know|consider|look for)\b|"
    r"\bwhy (?:you|your business|every business|businesses|small businesses) (?:need|should)\b|\bhere'?s (?:why|how|what)\b|"
    r"\bultimate guide\b|\bchecklist\b|\b(?:top|best)\s+\d+\b|\bwhat does an? [\w ]{3,40} do\b|\bhow much does\b|\bwhat it costs\b|"
    r"(?:\.\.\.|…)\s*but\b|\bbut your (?:job description|jd|budget)\b|\b\d+\+?\s+[\w ]{0,30}agencies\b|"
    r"\bbest [\w ]{0,30}(?:agencies|companies|tools|freelancers)\b|\bwhat (?:a|an) [\w ]{0,40} costs?\b|\bcost and\b|"
    r"\bpays? off\b|\b(?:shout|hit up|reach out to) (?:this guy|him|her|them)\b|\bhighly recommend (?:him|her|them|this)\b|"
    r"\bshortlist",
    re.I,
)

# -- growth signals ----------------------------------------------------------------------------

_BIZ_NOUN = (
    r"(?:gym|studio|salon|spa|med\s?spa|cafe|café|coffee\s+shop|bakery|restaurant|bar|bistro|boutique|shop|store|clinic|"
    r"practice|office|location|space|business|company|startup|brand|kitchen|barbershop|bakeshop|eatery|diner|pizzeria|"
    r"brewery|winery|hotel|venue|center|centre|academy|school|doors|app|platform|product|fitness)"
)
# Strong phrases: in a first-person sentence they are an announcement ("our grand opening").
_GROWTH_STRONG = re.compile(
    r"\b(?:grand opening|soft (?:opening|launch)|ribbon cutting|opening (?:day|week)|first day (?:open|of business)|"
    rf"just opened (?:up )?(?:our|my|the doors|a new|an?\s+(?:\w+\s+){{0,2}}?{_BIZ_NOUN})|opened (?:our|my) (?:\w+\s+){{0,2}}?{_BIZ_NOUN}|"
    rf"(?:just )?launched (?:our|my) (?:\w+\s+){{0,2}}?{_BIZ_NOUN}|launching (?:our|my) (?:\w+\s+){{0,2}}?{_BIZ_NOUN}|"
    r"(?:just |we |i )opened (?-i:[A-Z])|we(?:'| a)re opening\b|"
    r"(?:our|my) new (?:\w+\s+)?(?:location|studio|gym|salon|shop|store|cafe|restaurant|clinic)|"
    r"raised (?:\$|£|€)\d|(?:closed|raised) (?:our |a )?(?:seed|pre-?seed|series [abc])|seed round|pre-?seed round|"
    r"series [ab] (?:round|funding)|funding round)",
    re.I,
)
# Weak phrases also need a business noun in the sentence ("We're now open!" - but not "we're live on Twitch").
_GROWTH_WEAK = re.compile(
    r"\b(?:now open|opening soon|we(?:'| a)re (?:now |officially )?open|newly opened|we(?:'| a)re live|officially open)",
    re.I,
)
_FIRST_PERSON_RE = re.compile(r"\b(?:we|we're|we've|our|us|i|i'm|i've|my)\b", re.I)
_BIZ_NOUN_RE = re.compile(rf"\b{_BIZ_NOUN}\b", re.I)
_NOT_AN_OPENING_RE = re.compile(
    r"\b(?:applications?|nominations?|voting|votes?|registrations?|entries|submissions?|enrol?ments?|tickets?|grants?|"
    r"schemes?|calls?|rounds?|awards?|residenc(?:y|ies)|programm?e?s?|portal|sign-?ups?|pre-?orders?|bookings?|"
    r"waitlist|survey|polls?|competition|contest|challenge|hiring|positions?|roles?|vacanc(?:y|ies)|jobs?|cohort|"
    r"auditions?|casting|tryouts?|season)\b(?:\W+\w+){0,6}?\W+(?:is|are)?\s*(?:now\s+)?(?:officially\s+)?open\b|"
    r"\bclos(?:ing|ed|e) (?:our|the|its) doors\b|\bpermanently closed\b|\bclosing down\b|\bthat (?:just )?opened\b|"
    r"\banyone else\b|\bnot a grand opening\b",
    re.I,
)
# Someone celebrating another business's opening. Always: congratulations, "my friend", reviews.
_THIRD_PARTY_RE = re.compile(
    r"\bcongratulat\w*|\bmy (?:friend|friends|client|clients|sister|brother|cousin)\b|\bshow (?:them |some )?(?:love|support)\b|"
    r"\b(?:go |please )?check (?:them|it|her|him) out\b|\bi had the best time\b|\byour (?:favou?rite|local)\b|"
    r"\b(?:i|we) (?:went to|visited|attended|stopped by)\b|\bhad an? (?:awesome|amazing|great|wonderful|fantastic|blast)\b|"
    r"@\w+ (?:just )?opened\b|\blast (?:month|year|week)'?s\b|\b(?:one of )?(?:my|our) (?:very )?favou?rite\b|"
    r"\bspotlight\b|\bsummari[sz]ed by\b|\bheld (?:its|their) (?:grand|soft) opening\b|\bchamber of commerce\b|"
    r"\b(?:community|recreation|civic) (?:center|centre)\b|\bcity of\b|\breview\b|\bi rate\b|\bcelebrating \d+ years?\b|"
    r"\banniversary\b|\b(?:a )?(?:little )?(?:over )?(?:a|one|\d+) years? ago\b|\bplaybook\b|"
    r"\b(?:solution|software|platform|tool|trial|saas|api)\b|\bfor (?:local |small )?(?:businesses|operators|retailers)\b|"
    r"\bwe'?ve ever (?:built|opened)\b|\b(?:hq|headquarters)\b",
    re.I,
)
# "celebrate the grand opening of Montecito Med Spa": someone else's, unless the business is inviting people.
_OF_SOMEONE_RE = re.compile(r"\bthe grand opening of (?!our\b|my\b)", re.I)
_HOSTING_RE = re.compile(r"\bwe(?:'re| are) (?:celebrating|getting ready|hosting|throwing)\b|\bannounc\w* the grand opening\b", re.I)
# Unless the business itself speaks ("celebrate the grand opening of our new studio").
_THIRD_PARTY_WEAK_RE = re.compile(r"\bcelebrat\w* (?:the|their|his|her)\b|\bmarks\b|\bmilestones?\b", re.I)
# Invitations come from the business itself: "Join us for the grand opening of ...", "You're invited ..."
_INVITE_RE = re.compile(r"\bjoin us\b|\byou'?re invited\b|\bannounc\w* (?:the )?grand opening\b|\bcome (?:celebrate|see us|visit us|by)\b|\bsee you there\b", re.I)
_OWN_RE = re.compile(r"\b(?:our|we|we're|we've|my)\b", re.I)
# Big chains have their own marketing departments.
_KNOWN_CHAINS = re.compile(
    r"\b(?:anytime fitness|planet fitness|crunch(?: fitness)?|la fitness|orangetheory|f45|snap fitness|gold'?s gym|equinox|"
    r"24 hour fitness|vasa fitness|life time fitness|eos fitness|club pilates|pure barre|starbucks|mcdonald'?s|subway|"
    r"chipotle|domino'?s|pizza hut|kfc|burger king|wendy'?s|taco bell|dunkin|tim hortons|chick-fil-a|bojangles|"
    r"european wax center|massage envy|great clips|supercuts|sport clips|ulta|sephora|vio med spa|sola salons?|"
    r"amped fitness|hotworx|ey|deloitte|pwc|kpmg|85°c|ideal image|laser away|hand & stone|the joint)\b",
    re.I,
)
_CHAIN_RE = re.compile(
    r"\b(?:newest|latest)\s+(?:\w+\s+)?(?:location|club|branch|store|outlet|restaurant)\b|\b\d+(?:st|nd|rd|th)\s+(?:\w+\s+){0,3}?"
    r"(?:location|club|branch|store|outlet|restaurant|studio|gym)\b|#\d+\b|\b\d{2,}\s+(?:locations|clubs|stores|branches|"
    r"restaurants|studios|gyms|outlets)\b|\bfranchis\w*|\bour (?:newest|latest)\b|\bflagship\b|\bshowroom\b",
    re.I,
)

# -- scoring hints -----------------------------------------------------------------------------

_URGENT_RE = re.compile(r"\basap\b|urgent|immediately|right away|this week|start (now|asap|immediately)|as soon as", re.I)
_BUDGET_RE = re.compile(r"[$£€]\s?\d|budget|per month|/mo\b|retainer|paid (role|gig|position)|\brate\b", re.I)

# -- who is posting ------------------------------------------------------------------------------

# "my friend's gym", "my favorite cafe": someone else's business.
_NOT_MINE = (
    r"(?:friend|sister|brother|mom|mum|dad|mother|father|wife|husband|partner|client|boss|cousin|aunt|uncle|son|"
    r"daughter|neighbou?r|favou?rite|local|old|former)(?:'?s)?\s"
)
_OWNER_TITLE = r"(co-?founder|founder|co-?owner|owner|ceo|managing director|proprietor)"
_OWNER_RES = [
    # "I'm the owner", "I'm a co-founder of", "as CEO" - never "the owner's assistant"
    re.compile(rf"\b(?:i'?m|i am|as|i'?m one of)\s+(?:the\s+|a\s+|an\s+|one of the\s+)?{_OWNER_TITLE}(?!'s)\b", re.I),
    # "Owner of Glow Studio here", "Founder & CEO of Rosa Cafe"
    re.compile(rf"(?:^|[.!?\n]\s*){_OWNER_TITLE}(?:\s*(?:&|and)\s*\w+)?\s+(?:of|at)\s+\S", re.I),
    re.compile(r"\b(?:i'?m|i am)\s+(?:a\s+)?(?:small[- ])?business owner\b|\bbusiness owner here\b|\bfounder here\b", re.I),
    # "I own a bakery", "we run a small gym", "I started my salon"
    re.compile(rf"\b(?:i|we)\s+(?:own|run|founded|started|opened)\s+(?:a|an|the|my|our)?\s*(?:\w+\s+){{0,3}}?{_BIZ_NOUN}\b", re.I),
    # "my new pilates studio" - not "my friend's gym", and not "my company" (employees say that too)
    re.compile(
        r"\bmy\s+(?:(?!" + _NOT_MINE + r")[a-z'-]+\s+){0,2}?(gym|studio|business|shop|store|startup|salon|spa|restaurant|cafe|"
        r"clinic|practice|bakery|boutique)\b",
        re.I,
    ),
]
_HONORIFIC_RE = re.compile(r"^(?:dr|mr|mrs|ms|miss|mx|prof)\.?\s+", re.I)
_CREDENTIALS = {"mba", "phd", "md", "cpa", "pmp", "cfa", "esq", "jr", "sr", "ii", "iii", "iv", "dds", "dmd", "rn", "msc", "bsc"}
_NAME_WORD_RE = re.compile(r"^[^\W\d_][^\W\d_'’.\-]*(?:['’.\-][^\W\d_]+)*\.?$")
_BUSINESS_WORDS = re.compile(
    r"\b(gym|fitness|studio|spa|salon|cafe|coffee|bakery|clinic|dental|agency|shop|store|boutique|co|llc|inc|ltd|"
    r"group|labs?|club|kitchen|restaurant|bar|grill|yoga|pilates|crossfit|barbers?|beauty|wellness|realty|"
    r"homes|motors|media|digital|solutions|services|official|hq|team|company|brand|academy|school|center|centre)\b",
    re.I,
)
# Job aggregators repost other companies' openings; the poster is not the business.
_JOB_BOARDS = (
    "indeed.", "glassdoor.", "bebee.", "builtin.", "workindia.", "onlinejobs.ph", "wantapply.", "ziprecruiter.",
    "monster.", "simplyhired.", "careerbuilder.", "jooble.", "talent.com", "naukri.", "seek.com", "reed.co.uk",
    "totaljobs.", "cv-library.", "jobstreet.", "rozee.pk", "bayt.", "remoteok.", "weworkremotely.", "flexjobs.",
    "himalayas.app", "careerjet.", "adzuna.", "jora.", "edtech.com", "dice.com", "snagajob.", "workingnomads.",
    "fiverr.com", "jobs.", "careers.", "/jobs/",
)
# Applicant-tracking systems host a company's own job page: the company is the subdomain
# (acme.workable.com), so those are the business itself posting. Upwork-style job posts are clients hiring.
_ATS_HOSTS = ("workable.com", "jobsoid.com", "lever.co", "greenhouse.io", "breezy.hr", "recruitee.com", "ashbyhq.com",
              "bamboohr.com", "teamtailor.com", "personio.de", "personio.com", "jobvite.com", "smartrecruiters.com",
              "applytojob.com", "zohorecruit.com", "upwork.com", "freelancer.com", "peopleperhour.com", "guru.com")
_JOB_BOARD_TITLE = re.compile(r"jobs?\s+(?:in|near|at)|\d+\+?\s+[\w ]{0,40}jobs|jobs?\s*[|(-]|\(now hiring\)", re.I)
# Posters whose name or handle says they sell marketing, place staff, or list jobs.
_PROVIDER_LONG = ("agency", "marketing", "marketer", "digital", "creative", "advertising", "branding", "socialmedia",
                  "recruit", "staffing", "talent", "careers", "jobs", "hiring")
_PROVIDER_TOKENS = {"ads", "seo", "ppc", "smm", "media", "social", "growth", "content", "brand", "studios"}

# "we just opened Acme Fitness", "owner of Glow Med Spa", "founder at Launchly", "manager needed at Glow Spa".
# Trigger words in any case; the name itself must be capitalised. [ \t]: never across a line break.
_UPPER = r"[A-ZÀ-ÖØ-ÞĀ-Ž]"
_BUSINESS_RE = re.compile(
    r"\b(?i:opened|launched|launching|owner of|co-?owner of|founder of|co-?founder of|ceo of|founder at|ceo at|owner at|"
    r"run|own|(?:needed|wanted|hiring|position|role|job|manager|owner|founder|ceo) at)[ \t]+"
    rf"((?:{_UPPER}[\w&'’]*)(?:[ \t]+(?:{_UPPER}[\w&'’]*|&|of)){{0,4}})"
)
# Capitalised words that start sentences, name platforms or are marketing jargon, not businesses.
_NOT_A_NAME = frozenset(
    """facebook instagram tiktok linkedin google meta youtube twitter x reddit pinterest snapchat shopify etsy amazon
    threads whatsapp monday tuesday wednesday thursday friday saturday sunday today tomorrow tonight january february
    march april may june july august september october november december the a an our my this that we i it its
    new last next ads ad shop seo ppc ugc smm sem crm version q1 q2 q3 q4 black cyber campaign campaigns reels stories
    content marketing social media sales""".split()
)

# -- websites ------------------------------------------------------------------------------------

# Lookbehinds anchor each match at a token start, so long runs of dots/letters stay linear.
_EMAIL_RE = re.compile(r"(?<![\w.%+-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}")
_URL_RE = re.compile(
    r"(?<![\w.@/-])(https?://)?(www\.)?([a-z0-9][a-z0-9-]{0,62}(?:\.[a-z0-9-]{1,62})*\.([a-z]{2,24}))(?![\w@-])(/[^\s]*)?",
    re.I,
)
# Real TLDs a small business might use. Two-letter country codes are accepted too, except the
# ones that are really file or language extensions in running text (next.js, setup.py, README.md).
_GTLDS = frozenset(
    """com net org info biz io ai app dev co shop store online site website tech studio fitness salon spa cafe
    restaurant coffee health clinic dental beauty agency marketing media design academy club live life world today
    space xyz pro me tv bar pizza kitchen boutique fashion style yoga gym coach consulting solutions services
    company business email global group digital social network events photography photo gallery art
    church school education law legal realty homes properties house rentals travel tours vegas nyc london""".split()
)
_NOT_TLDS = frozenset({"js", "py", "ts", "md", "rb", "sh", "db", "cs"})

# -- locations -----------------------------------------------------------------------------------

_US_STATES = """alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho
illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri
montana nebraska nevada new\\ hampshire new\\ jersey new\\ mexico new\\ york north\\ carolina north\\ dakota ohio oklahoma
oregon pennsylvania rhode\\ island south\\ carolina south\\ dakota tennessee texas utah vermont virginia washington
west\\ virginia wisconsin wyoming""".split()
_PLACES: dict[str, str] = {
    "united states": "United States", "usa": "United States", "u.s.": "United States", "america": "United States",
    "nyc": "United States", "los angeles": "United States", "chicago": "United States", "houston": "United States",
    "austin": "United States", "dallas": "United States", "miami": "United States", "atlanta": "United States",
    "boston": "United States", "seattle": "United States", "denver": "United States", "phoenix": "United States",
    "san francisco": "United States", "san diego": "United States", "las vegas": "United States",
    "canada": "Canada", "toronto": "Canada", "vancouver": "Canada", "montreal": "Canada", "calgary": "Canada",
    "ottawa": "Canada", "united kingdom": "United Kingdom", "uk": "United Kingdom", "england": "United Kingdom",
    "scotland": "United Kingdom", "wales": "United Kingdom", "britain": "United Kingdom", "london": "United Kingdom",
    "manchester": "United Kingdom", "birmingham": "United Kingdom", "leeds": "United Kingdom", "glasgow": "United Kingdom",
    "australia": "Australia", "sydney": "Australia", "melbourne": "Australia", "brisbane": "Australia", "perth": "Australia",
    "new zealand": "New Zealand", "auckland": "New Zealand", "ireland": "Ireland", "dublin": "Ireland",
    "india": "India", "mumbai": "India", "delhi": "India", "bangalore": "India", "bengaluru": "India", "thane": "India",
    "pune": "India", "hyderabad": "India", "chennai": "India", "kolkata": "India", "pakistan": "Pakistan",
    "karachi": "Pakistan", "lahore": "Pakistan", "islamabad": "Pakistan", "bangladesh": "Bangladesh", "dhaka": "Bangladesh",
    "nigeria": "Nigeria", "lagos": "Nigeria", "abuja": "Nigeria", "kenya": "Kenya", "nairobi": "Kenya",
    "south africa": "South Africa", "johannesburg": "South Africa", "cape town": "South Africa", "ghana": "Ghana",
    "accra": "Ghana", "egypt": "Egypt", "cairo": "Egypt", "uae": "United Arab Emirates", "dubai": "United Arab Emirates",
    "abu dhabi": "United Arab Emirates", "saudi arabia": "Saudi Arabia", "riyadh": "Saudi Arabia", "qatar": "Qatar",
    "philippines": "Philippines", "manila": "Philippines", "indonesia": "Indonesia", "jakarta": "Indonesia",
    "malaysia": "Malaysia", "kuala lumpur": "Malaysia", "singapore": "Singapore", "germany": "Germany", "berlin": "Germany",
    "munich": "Germany", "france": "France", "paris": "France", "spain": "Spain", "madrid": "Spain", "barcelona": "Spain",
    "italy": "Italy", "milan": "Italy", "rome": "Italy", "netherlands": "Netherlands", "amsterdam": "Netherlands",
    "belgium": "Belgium", "sweden": "Sweden", "stockholm": "Sweden", "denmark": "Denmark", "copenhagen": "Denmark",
    "norway": "Norway", "oslo": "Norway", "finland": "Finland", "poland": "Poland", "portugal": "Portugal",
    "lisbon": "Portugal", "brazil": "Brazil", "são paulo": "Brazil", "sao paulo": "Brazil", "mexico": "Mexico",
    "argentina": "Argentina", "colombia": "Colombia", "chile": "Chile", "peru": "Peru", "japan": "Japan", "tokyo": "Japan",
    "china": "China", "hong kong": "Hong Kong", "south korea": "South Korea", "seoul": "South Korea", "israel": "Israel",
    "tel aviv": "Israel", "turkey": "Turkey", "istanbul": "Turkey",
}
_PLACES.update({s.replace("\\", ""): "United States" for s in _US_STATES})
_PLACE_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(p) for p in _PLACES), key=len, reverse=True)) + r")\b", re.I)
_US_ABBREV_RE = re.compile(
    r",\s*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|"
    r"OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b"
)
_CCTLD = {"uk": "United Kingdom", "ca": "Canada", "au": "Australia", "nz": "New Zealand", "ie": "Ireland", "in": "India",
          "pk": "Pakistan", "bd": "Bangladesh", "ng": "Nigeria", "ke": "Kenya", "za": "South Africa", "gh": "Ghana",
          "eg": "Egypt", "ae": "United Arab Emirates", "sa": "Saudi Arabia", "qa": "Qatar", "ph": "Philippines",
          "id": "Indonesia", "my": "Malaysia", "sg": "Singapore", "de": "Germany", "fr": "France", "es": "Spain",
          "it": "Italy", "nl": "Netherlands", "be": "Belgium", "se": "Sweden", "dk": "Denmark", "no": "Norway",
          "fi": "Finland", "pl": "Poland", "pt": "Portugal", "br": "Brazil", "mx": "Mexico", "ar": "Argentina",
          "co": "Colombia", "cl": "Chile", "pe": "Peru", "jp": "Japan", "cn": "China", "hk": "Hong Kong", "kr": "South Korea",
          "il": "Israel", "tr": "Turkey", "us": "United States"}
_TARGET_ALIASES = {"us": "United States", "u.s.": "United States", "usa": "United States", "america": "United States",
                   "uk": "United Kingdom", "great britain": "United Kingdom", "uae": "United Arab Emirates"}

# (pattern, niche). Word starts only: "coworking space" is not a spa, "happy" is not an app.
_NICHE_WORDS = [
    (r"gyms?|fitness|pilates|yoga|crossfit", "fitness"),
    (r"med ?spas?|spas?|salons?|barbers?(?:hop)?|nails?|beauty", "beauty"),
    (r"restaurants?|cafes?|coffee|bakery|bakeries", "food & drink"),
    (r"clinics?|dental|dentists?|chiropract\w*", "healthcare"),
    (r"startups?|saas|apps?", "startup"),
    (r"e-?commerce|shopify", "e-commerce"),
    (r"real estate|realtors?", "real estate"),
    (r"law firms?", "legal"),
    (r"boutiques?", "retail"),
]
_NICHE_RES = [(re.compile(rf"\b(?:{pat})\b", re.I), label) for pat, label in _NICHE_WORDS]


# -- helpers ---------------------------------------------------------------------------------------


def _quotes(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?\n])\s+|\n+", text) if s.strip()]


def _services(text: str) -> list[ServiceKind]:
    t = text.lower()
    out: list[ServiceKind] = []
    if re.search(r"social\s*media|socials|smm|instagram|\big\b|tiktok", t):
        out.append("social_media_management")
    if re.search(r"\bads?\b|ppc|media\s*buyer|paid", t):
        out.append("paid_ads_management")
    if re.search(r"content|ugc|creator|video|reels", t):
        out.append("content_creation")
    return out or ["general_digital_marketing"]


def _request(text: str) -> Optional[str]:
    """The sentence asking for help, if any. Bare questions ("Need a social media manager?") are
    pitch hooks, not requests; questions asking for names ("Anyone recommend...?") are requests."""
    found = None
    switch = _SWITCH_RE.search(text)
    if switch:
        return switch.group(0).strip()
    for sentence in _sentences(text):
        m = _REQUEST_RE.search(sentence) or _WANTED_RE.search(sentence) or _SOMEONE_RE.search(sentence) \
            or _RECOMMEND_AFTER_RE.search(sentence)
        # The thing asked for is a course or a tool only if that word comes before or right after the service
        # ("a course on Facebook ads", "a social media management tool"), not "promote my new app" later on.
        if not m or _NOT_A_SERVICE_RE.search(sentence[: m.end() + 20]):
            continue
        if re.search(r"\bif (?:you|anyone|anybody|your (?:business|brand|company))(?:'re| are| is)?\s+(?:\w+\s+)?$",
                     sentence[: m.start()], re.I):
            continue  # "If you're looking for an ads manager, call this guy" - advice, not a request
        rest = sentence[m.end():]
        # The '?' must close the request itself; "...Expert | Remote | Can you turn ad spend into sales?" is a new clause.
        q = re.match(r"([^.!?|\n]{0,90})\?", rest)
        is_question = bool(q) and not re.search(r"\b(?:can|could|do|are|will|would) you\b", q.group(1), re.I)
        if is_question and not _RECOMMEND.search(sentence):
            return None  # "Need a social media manager? ..." - the whole post is a pitch
        found = found or sentence.strip()
    return found


def _headline(post: RawPost) -> str:
    """Title plus the opening of the text, where a content post shows itself ("5 signs...", "How to hire...")."""
    body = re.sub(r"^\s*(?:\w{3} \d{1,2}, \d{4}|\d+ (?:hours?|days?|weeks?) ago)\s*(?:\.\.\.|·)?\s*", "", post.text or "")
    return _quotes(f"{post.title or ''}\n{body[:120]}")


_HIRER_RE = re.compile(r"((?:[A-Z][\w&'’.-]*[ \t]?){1,4}?)[ \t]+(?:is|are)[ \t]+(?:now[ \t]+)?(?:looking for|hiring|seeking)\b")


_FORMER_RE = re.compile(r"\b(?:fired|left|dropped|let go of|leaving|parted ways with|done with|unhappy with|switching from|"
                        r"previous|current|old|former|ex-?)\W+(?:\w+\W+){0,1}$", re.I)


def _seller(text: str) -> bool:
    if SELLER_PATTERNS.search(text) or _SELLER_CLAIM.search(text):
        return True
    for m in _SELLER_STRONG.finditer(text):
        if not _FORMER_RE.search(text[max(0, m.start() - 30): m.start()]):
            return True
    # "Socialander is looking for a Social Media Manager": a marketing company hiring for its clients
    return any(_is_provider_name(m.group(1)) or re.search(r"social|media|digital|agency|advertis|marketing", m.group(1), re.I)
               for m in _HIRER_RE.finditer(text))


def _job_seeker(text: str) -> bool:
    return bool(_JOB_SEEKER_RE.search(text)) and not _HIRING_CUE.search(text)


def _poster_names(post: RawPost) -> list[str]:
    """Who wrote the post: page metadata, the URL handle, and the 'Name | ...' / "Name's Post" title prefix."""
    names = [post.author or ""]
    m = re.search(r"linkedin\.com/posts/([^_/?]+)", post.url) or re.search(r"(?:x|twitter)\.com/([^/?]+)/status", post.url) \
        or re.search(r"facebook\.com/([^/?]+)/posts", post.url)
    if m:
        names.append(m.group(1))
    host = re.sub(r"^https?://(?:www\.)?", "", post.url).split("/")[0].lower()
    ats = next((a for a in _ATS_HOSTS if host == a or host.endswith("." + a)), None)
    if ats:
        if host != ats:
            names.append(host[: -len(ats) - 1])  # acme.workable.com -> acme
    elif post.platform == "web" and host:
        names.append(host.rsplit(".", 1)[0])  # skywaymedia.com: the site owner
    if post.platform == "instagram" and " | " in (post.title or ""):
        names.append(post.title.split(" | ", 1)[0])
    m = re.search(r"\b([a-z0-9_.]{3,40}) on (?:[A-Z][a-z]+ \d{1,2}, \d{4})", post.text or "")
    if m:
        names.append(m.group(1))  # "zestfitnessstudio on October 2, 2026:" - the Instagram account
    # Titles name the author only in fixed forms: "Jane Doe's Post", "Glow Studio on Instagram: ...".
    m = re.match(r"\s*(.{2,60}?)(?:['’]s? Post\b| on (?:X|Instagram|LinkedIn|Facebook|Threads)\b| posted on\b)", post.title or "")
    if m:
        names.append(m.group(1))
    return [n for n in names if n]


def _is_provider_name(name: str) -> bool:
    flat = re.sub(r"[^a-z]", "", name.lower())
    tokens = set(re.split(r"[^a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()))
    return any(w in flat for w in _PROVIDER_LONG) or bool(tokens & _PROVIDER_TOKENS)


def _person(author: str) -> Optional[str]:
    """'Dr. José Núñez, MBA | Founder 🌸' -> 'José Núñez'. None for business pages and handles."""
    name = re.split(r"\s*(?:[,|•·–—]|\s-\s)\s*", author, maxsplit=1)[0]
    name = re.sub(r"\([^)]*\)", " ", name)
    name = _HONORIFIC_RE.sub("", " ".join(w for w in name.split() if re.search(r"[^\W\d_]", w)))
    words = [w for w in name.split() if w.lower().strip(".") not in _CREDENTIALS]
    if not 2 <= len(words) <= 4 or _BUSINESS_WORDS.search(" ".join(words)):
        return None
    if not all(_NAME_WORD_RE.match(w) and w[0].isupper() for w in words):
        return None
    return " ".join(words)


_COMPOUND_BIZ = ("coffee", "cafe", "bakery", "bakeshop", "fitness", "studio", "salon", "spa", "gym", "kitchen", "shop",
                 "store", "clinic", "dental", "boutique", "pilates", "yoga", "barber", "beauty", "wellness", "realty", "homes",
                 "group", "labs", "brewing", "roasters", "pizza", "grill", "bistro", "eatery", "academy")


def _looks_like_business(name: str) -> bool:
    """'Glow Studio', 'GlowStudio' and the joined-up handle 'downeastcoffee' all name a business."""
    if _BUSINESS_WORDS.search(re.sub(r"([a-z])([A-Z])", r"\1 \2", name)):
        return True
    token = name.strip().lower()
    return " " not in token and len(token) >= 6 and any(w in token for w in _COMPOUND_BIZ)


def _owner(text: str) -> tuple[bool, Optional[str]]:
    for pattern in _OWNER_RES:
        m = pattern.search(text)
        if m:
            title = next((g for g in m.groups() if g), None) if m.groups() else None
            if title and title.lower() in ("ceo",):
                return True, "CEO"
            if title and re.fullmatch(_OWNER_TITLE, title, re.I):
                return True, title.title()
            return True, "Owner"
    return False, None


def _growth(text: str, business_poster: bool) -> bool:
    """A sentence where the business itself announces its opening, launch or funding."""
    if _CHAIN_RE.search(text) or _KNOWN_CHAINS.search(text) or _THIRD_PARTY_RE.search(text):
        return False
    if _OF_SOMEONE_RE.search(text) and not (_INVITE_RE.search(text) or _HOSTING_RE.search(text)):
        return False
    post_speaker = bool(_FIRST_PERSON_RE.search(text) or _INVITE_RE.search(text))
    post_noun = bool(_BIZ_NOUN_RE.search(text))
    for sentence in _sentences(text):
        if _NOT_AN_OPENING_RE.search(sentence):
            continue
        if _THIRD_PARTY_WEAK_RE.search(sentence) and not _OWN_RE.search(sentence):
            continue
        speaker = bool(_FIRST_PERSON_RE.search(sentence) or _INVITE_RE.search(sentence)) or business_poster
        if _GROWTH_STRONG.search(sentence) and (speaker or _BIZ_NOUN_RE.search(sentence) or post_noun):
            return True  # "GRAND OPENING - ZEST FITNESS STUDIO"
        if _GROWTH_WEAK.search(sentence) and (speaker or post_speaker) and (_BIZ_NOUN_RE.search(sentence) or business_poster):
            return True  # "SYLVA HEALTH & FITNESS IS NOW OPEN! We're excited ..."
    return False


_EN_WORDS = frozenset("the and to of a in for is our we you with on at this are be it my i an your looking need hiring".split())
# Function words of the other languages that show up in these searches (Nordic, Spanish, German, French, Dutch, Portuguese).
_FOREIGN_WORDS = frozenset(
    """og af på ikke vores hos jeg eller mit fra vil nye blev efter vi har det er til som var
    el los las una para con por que del nuestro nuestra está estamos y
    und der die das nicht mit wir unser unsere ist für auf ein eine
    le la les des une pour avec nous notre est dans sur et
    het een niet wij onze voor van
    os uma não nós nosso nossa com""".split()
)


def _english(text: str) -> bool:
    """The rules only understand English: 'I går var det grand opening hos ...' is not judged.
    Snippets full of hashtags and abbreviations still count as English."""
    words = re.findall(r"[^\W\d_']+", text.lower())
    foreign = sum(w in _FOREIGN_WORDS for w in words)
    return foreign < 3 or foreign <= sum(w in _EN_WORDS for w in words)


# "Jive is hiring a social media manager", "hiring a social media manager for Luckin Coffee"
_HIRING_FOR_RE = re.compile(
    rf"\b(?i:hiring|looking for|seeking)\b[^.\n]{{0,50}}?\b(?i:at|for)[ \t]+((?:{_UPPER}[\w&'’]*)(?:[ \t]+(?:{_UPPER}[\w&'’]*|&)){{0,3}})"
)


def _business_name(text: str) -> Optional[str]:
    for m in [*_BUSINESS_RE.finditer(text), *_HIRER_RE.finditer(text), *_HIRING_FOR_RE.finditer(text)]:
        words = m.group(1).split()
        while words and words[0].lower().strip("'") in _NOT_A_NAME:
            words.pop(0)
        while words and words[-1] in ("&", "of"):
            words.pop()
        if not words or all(w.lower().strip("'") in _NOT_A_NAME for w in words):
            continue
        if len(words) == 1 and _SERVICE_RE.fullmatch(words[0]):
            continue
        if words[0] in ("We", "I", "They", "He", "She", "It", "Who", "Everyone", "Anyone", "Someone"):
            continue
        return " ".join(words)
    return None


def _website(text: str, business: Optional[str] = None) -> Optional[str]:
    """The business's own site: skips social, booking and builder hosts, emails, files and
    joined sentences ("Austin.We"); prefers a domain that matches the business name."""
    found: list[str] = []
    for m in _URL_RE.finditer(text):
        scheme, www, host, tld = m.group(1), m.group(2), m.group(3), m.group(4)
        if m.end() < len(text) and text[m.end()] == "@":  # the name part of an email address
            continue
        if not (scheme or www) and tld[0].isupper() and not host.isupper():
            continue  # "in Austin.We need" - a missing space, not a domain
        tld = tld.lower()
        if tld in _NOT_TLDS or (len(tld) > 2 and tld not in _GTLDS):
            continue  # next.js, resume.pdf
        domain = normalize_domain(host.lower())
        if domain and domain not in found:
            found.append(domain)
    if business:
        for domain in found:
            if name_matches(business, domain):
                return domain
    return found[0] if found else None


def _location(text: str, url: str, website: Optional[str]) -> Optional[str]:
    m = _PLACE_RE.search(text)
    if m:
        return _PLACES[m.group(1).lower()]
    if _US_ABBREV_RE.search(text):
        return "United States"
    for host in (website or "", re.sub(r"^https?://", "", url).split("/")[0]):
        parts = host.lower().split(".")
        if host.endswith("linkedin.com") and len(parts) == 3 and len(parts[0]) == 2 and parts[0] != "www":
            return _CCTLD.get(parts[0])  # ng.linkedin.com: the poster's country
        if website and host == website and len(parts[-1]) == 2 and parts[-1] in _CCTLD:
            return _CCTLD[parts[-1]]
    return None


def _outside_targets(location: Optional[str], targets: list[str]) -> bool:
    if not location or not targets:
        return False
    wanted = set()
    for t in targets:
        key = t.strip().lower()
        wanted.add(_TARGET_ALIASES.get(key) or _PLACES.get(key) or t.strip())
    return location not in wanted


def _niche(text: str, post: RawPost) -> Optional[str]:
    if post.niche:
        return post.niche
    return next((label for pattern, label in _NICHE_RES if pattern.search(text)), None)


# -- the classifier ------------------------------------------------------------------------------


def assess_without_ai(post: RawPost, cfg: Config) -> IntentAssessment:
    text = _quotes(post.content())
    author = _quotes(post.author or "").strip()
    person = _person(author)
    business_poster = bool(author) and not person and not author.startswith("u/") and _looks_like_business(author)

    names = _poster_names(post)
    business_poster = business_poster or any(not _person(n) and _looks_like_business(n) for n in names[1:])
    url = post.url.lower()
    job_board = any(b in url for b in _JOB_BOARDS) or bool(_JOB_BOARD_TITLE.search(post.title or ""))
    seller = _seller(text) or job_board or any(_is_provider_name(n) for n in names) or not _english(text)
    job_seeker = _job_seeker(text)
    content = bool(_CONTENT_RE.search(_headline(post)))
    request = None if (job_seeker or content) else _request(text)
    growth = bool(post.niche) and not seller and _growth(text, business_poster)

    business = (
        post.extra.get("business_name")
        or _business_name(text)
        or (author if business_poster else None)
    )
    if business and _is_provider_name(business):
        seller = True  # "We're hiring a social media manager at RMD Advertising": an agency hiring staff
    website = post.extra.get("website") or _website(text, business)
    location = _location(text, post.url, website)

    if request and not seller:
        intent_type = "explicit_request"
        score = 72 + 10 * bool(_URGENT_RE.search(text)) + 8 * bool(_BUDGET_RE.search(text)) + 5 * bool(website)
    elif growth:
        intent_type, score = "growth_signal", 78
    else:
        intent_type, score = "not_relevant", 20 if _SERVICE_RE.search(text) else 5
    if intent_type != "not_relevant" and _outside_targets(location, cfg.agency.target_locations):
        intent_type, score = "not_relevant", 10  # the agency does not serve this location

    is_owner, role = _owner(text)
    emails = list(dict.fromkeys(e for e in _EMAIL_RE.findall(text)))
    return IntentAssessment(
        intent_type=intent_type,
        author_is_service_provider=seller or job_seeker,
        intent_score=min(score, 100),
        services_needed=_services(text),
        business_name=business,
        business_website=website,
        business_location=location,
        niche=_niche(text, post),
        poster_name=person,
        poster_role=role,
        poster_is_decision_maker=is_owner,
        emails_in_post=emails,
        summary=(request or text[:160]).strip(),
    )
