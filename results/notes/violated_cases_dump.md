# Cas VIOLATED -- dump complet pour inspection qualitative

Pour chaque terme VIOLATED : la cible, le terme, la citation qui a justifie la precondition, les labels d'activite BPMN matches (avec score) des deux cotes, et les aretes du graphe process touchant ces etats. Objectif : trancher 'vrai desordre' vs 'artefact de matching' cas par cas.


## Modele : llama-3.1-8b

### E_j02/0.bpmn2.xml
- cible   : `model.selected`  <- activite matchee : "select a parent type" (score=0.3794)
- terme   : `maternity_leave.planned`  <- activite matchee : "The employee approve parental leave request" (score=0.4603)
- operateur : AND  |  citation : "Let parent select"
- aretes process touchant ces etats (6) :
    - information.collected -> maternity_leave.planned ("Social Security sends back Parents Information to employee" -> "The employee approve parental leave request")
    - model.selected -> information.collected ("select a parent type" -> "send all the neccesarry Info  to employer")
    - model.selected -> maternity_leave.taken ("select a parent type" -> "take  protection period ")
    - maternity_leave.planned -> information.collected ("The employee approve parental leave request" -> "Employee sends parent information to Social Security")
    - maternity_leave.planned -> information.gathered ("The employee approve parental leave request" -> "notify to the parent")
    - model.selected -> model.selected ("choose the employment type" -> "select a parent type")

### E_j02/0.bpmn2.xml
- cible   : `model.selected`  <- activite matchee : "select a parent type" (score=0.3794)
- terme   : `information.collected`  <- activite matchee : "Social Security receives Info und updates the data" (score=0.3842)
- operateur : AND  |  citation : "Let parent select"
- aretes process touchant ces etats (17) :
    - information.gathered -> information.collected ("notify to the employer" -> "check the if the request informed in last 3 month")
    - information.collected -> maternity_leave.taken ("Social Security sends back Parents Information to employee" -> "The employee reject parental leave request")
    - information.collected -> maternity_leave.planned ("Social Security sends back Parents Information to employee" -> "The employee approve parental leave request")
    - information.collected -> information.gathered ("send all the neccesarry Info  to employer" -> "The employer receives info and checks  it")
    - information.collected -> information.collected ("Employee sends parent information to Social Security" -> "Social Security receives Info und updates the data")
    - information.collected -> information.collected ("Social Security checks the Parents Information" -> "Social Security sends back Parents Information to employee")
    - model.selected -> information.collected ("select a parent type" -> "send all the neccesarry Info  to employer")
    - model.selected -> maternity_leave.taken ("select a parent type" -> "take  protection period ")
    - maternity_leave.taken -> information.collected ("The employee reject parental leave request" -> "Employee sends parent information to Social Security")
    - information.collected -> maternity_leave.extended ("check the if the request informed in last 3 month" -> "reject the extension")
    - information.collected -> maternity_leave.extended ("check the if the request informed in last 3 month" -> "extend the parental leave")
    - maternity_leave.planned -> information.collected ("The employee approve parental leave request" -> "Employee sends parent information to Social Security")
    - ... (+5)

### G_g03/1.bpmn2.xml
- cible   : `battle_net_account.existing`  <- activite matchee : "Create new battle.net account" (score=0.6954)
- terme   : `battle_net_account.active`  <- activite matchee : "Log into game" (score=0.3534)
- operateur : AND  |  citation : "If you do not have one yet, you enter the account information and click the link you receive in the confirmation mail. As soon as you have a battle.net account, you can check if you have an active WoW subscription."
- aretes process touchant ces etats (9) :
    - credit_card.entered -> battle_net_account.active ("Enter credit card information" -> "Log into game")
    - battle_net_account.existing -> character_name.confirmed ("Create new battle.net account" -> "Receive email with confirmation link")
    - battle_net_account.active -> realm.selected ("Log into game" -> "Select realm of your character")
    - battle_net_account.active -> race.selected ("Log into game" -> "Select class of your character")
    - battle_net_account.active -> race.selected ("Log into game" -> "Select race of your character")
    - subscription.active -> battle_net_account.active ("Check WoW subscription" -> "Log into game")
    - battle_net_account.existing -> battle_net_account.existing ("Check existence of battle.net account" -> "Create new battle.net account")
    - battle_net_account.existing -> subscription.active ("Check existence of battle.net account" -> "Check WoW subscription")
    - bank_account.selected -> battle_net_account.active ("Enter bank account information" -> "Log into game")

### G_g03/1.bpmn2.xml
- cible   : `battle_net_account.existing`  <- activite matchee : "Create new battle.net account" (score=0.6954)
- terme   : `subscription.active`  <- activite matchee : "Check WoW subscription" (score=0.4655)
- operateur : AND  |  citation : "If you do not have one yet, you enter the account information and click the link you receive in the confirmation mail. As soon as you have a battle.net account, you can check if you have an active WoW subscription."
- aretes process touchant ces etats (6) :
    - battle_net_account.existing -> character_name.confirmed ("Create new battle.net account" -> "Receive email with confirmation link")
    - account.checked -> subscription.active ("Confirm account information via email link" -> "Check WoW subscription")
    - subscription.active -> battle_net_account.active ("Check WoW subscription" -> "Log into game")
    - subscription.active -> payment_method.selected ("Check WoW subscription" -> "Select payment method")
    - battle_net_account.existing -> battle_net_account.existing ("Check existence of battle.net account" -> "Create new battle.net account")
    - battle_net_account.existing -> subscription.active ("Check existence of battle.net account" -> "Check WoW subscription")

### G_g03/5.bpmn2.xml
- cible   : `payment_method.selected`  <- activite matchee : "Select a Payment Method" (score=0.8575)
- terme   : `battle_net_account.active`  <- activite matchee : "log into the game" (score=0.354)
- operateur : AND  |  citation : "If not, you can select the payment method."
- aretes process touchant ces etats (9) :
    - message.received -> battle_net_account.active ("Click the Confirmation Link" -> "log into the game")
    - message.received -> payment_method.selected ("Click the Confirmation Link" -> "Select a Payment Method")
    - credit_card.entered -> battle_net_account.active ("Enter Credit Card Information" -> "log into the game")
    - bank_account.entered -> battle_net_account.active ("Setting Up your Account" -> "log into the game")
    - bank_account.entered -> payment_method.selected ("Setting Up your Account" -> "Select a Payment Method")
    - battle_net_account.active -> realm.selected ("log into the game" -> "Select a Realm, Race, Class of your Charachter")
    - iban.entered -> battle_net_account.active ("Enter your IBAN and BIC numbers" -> "log into the game")
    - payment_method.selected -> credit_card.entered ("Select a Payment Method" -> "Enter Credit Card Information")
    - payment_method.selected -> iban.entered ("Select a Payment Method" -> "Enter your IBAN and BIC numbers")

### G_g03/5.bpmn2.xml
- cible   : `bank_account.entered`  <- activite matchee : "Enter Account Information" (score=0.6269)
- terme   : `battle_net_account.active`  <- activite matchee : "log into the game" (score=0.354)
- operateur : AND  |  citation : "If you choose your bank account, enter your IBAN and BIC numbers."
- aretes process touchant ces etats (8) :
    - message.received -> battle_net_account.active ("Click the Confirmation Link" -> "log into the game")
    - bank_account.entered -> message.received ("Enter Account Information" -> "Getting Confirmation Email")
    - credit_card.entered -> battle_net_account.active ("Enter Credit Card Information" -> "log into the game")
    - bank_account.entered -> bank_account.entered ("Setting Up your Account" -> "Enter Account Information")
    - bank_account.entered -> battle_net_account.active ("Setting Up your Account" -> "log into the game")
    - bank_account.entered -> payment_method.selected ("Setting Up your Account" -> "Select a Payment Method")
    - battle_net_account.active -> realm.selected ("log into the game" -> "Select a Realm, Race, Class of your Charachter")
    - iban.entered -> battle_net_account.active ("Enter your IBAN and BIC numbers" -> "log into the game")

### G_g03/5.bpmn2.xml
- cible   : `selfies.received`  <- activite matchee : "Get confirmation and selfies of your character" (score=0.5143)
- terme   : `expansion.released`  <- activite matchee : "Get the Message that the Expansion is released" (score=0.6391)
- operateur : AND  |  citation : "You get a confirmation, and some selfies of your character, as soon as a expansion is released you get another message."
- aretes process touchant ces etats (2) :
    - selfies.received -> expansion.released ("Get confirmation and selfies of your character" -> "Get the Message that the Expansion is released")
    - character_name.available -> selfies.received ("Enter name for the Character" -> "Get confirmation and selfies of your character")

### R_j02/2.bpmn2.xml
- cible   : `machine.inspected`  <- activite matchee : "Start inspection" (score=0.4784)
- terme   : `questions.asked`  <- activite matchee : "Ask additional questions" (score=0.7164)
- operateur : AND  |  citation : "Questions are asked, and you have to input values."
- aretes process touchant ces etats (8) :
    - questions.asked -> values.input ("Ask additional questions" -> "Input value(s)")
    - questions.asked -> questions.asked ("Display questions" -> "Q1: Filling volume?")
    - questions.asked -> questions.asked ("Display questions" -> "Q2: Residual liquid?")
    - questions.asked -> values.input ("Q2: Residual liquid?" -> "Input value")
    - questions.asked -> values.input ("Q1: Filling volume?" -> "Input value")
    - values.shown -> questions.asked ("Display values" -> "Ask additional questions")
    - questions.asked -> questions.asked ("Open questionnaire form" -> "Display questions")
    - machine.inspected -> questions.asked ("Start inspection" -> "Open questionnaire form")

### R_j02/8.bpmn2.xml
- cible   : `machine.inspected`  <- activite matchee : " Type of Machine" (score=0.465)
- terme   : `values.inputted`  <- activite matchee : "Input Values" (score=0.7607)
- operateur : AND  |  citation : "Questions are asked, and you have to input values."
- aretes process touchant ces etats (8) :
    - values.inputted -> values.inputted ("Input Values" -> "Show Values automatically")
    - machine.inspected -> machine.inspected ("Serial Number of Machine" -> "Start Inspection")
    - machine.inspected -> machine.inspected (" Type of Machine" -> "Start Inspection")
    - values.inputted -> values.inputted ("Show Values automatically" -> "Input Values")
    - machine.inspected -> values.collected ("Start Inspection" -> "Unsuccessfull Result")
    - machine.inspected -> values.inputted ("Start Inspection" -> "Input Values")
    - machine.inspected -> machine.inspected ("Please enter Serial Number and Type of Machine" -> "Serial Number of Machine")
    - machine.inspected -> machine.inspected ("Please enter Serial Number and Type of Machine" -> " Type of Machine")

### V_k09/2.bpmn2.xml
- cible   : `product.packed`  <- activite matchee : "Pack all items" (score=0.5234)
- terme   : `order.ready_for_shipment`  <- activite matchee : "Ship the order" (score=0.5892)
- operateur : AND  |  citation : "If the order is ready for shipment, a courier is requested and the products are packed simultaneously and finally shipped."
- aretes process touchant ces etats (4) :
    - product.packed -> order.ready_for_shipment ("Pack all items" -> "Ship the order")
    - order.checked -> order.ready_for_shipment ("Check the order" -> "Request a courier")
    - order.checked -> product.packed ("Check the order" -> "Pack all items")
    - order.ready_for_shipment -> order.ready_for_shipment ("Request a courier" -> "Ship the order")


## Modele : mistral-nemotron

### E_j03/2.bpmn2.xml
- cible   : `work_accident.under_review`  <- activite matchee : "check accident site" (score=0.4657)
- terme   : `work_accident.reported`  <- activite matchee : "Report accident insurance" (score=0.533)
- operateur : AND  |  citation : "Independent of this, every work accident in which a person with accident insurance has been killed, or injured in such a way that they are unable to work for three days, in full or in part, must be reported to the responsible accident insurance provider within five days."
- aretes process touchant ces etats (8) :
    - self_employed_person.reported -> work_accident.reported ("report employer " -> "Report accident insurance")
    - self_employed_person.reported -> work_accident.reported ("report employer " -> "Inform work inspectorate")
    - work_accident.reported -> doctor_visit.scheduled ("Inform work inspectorate" -> "merge documents")
    - private_insurance.reported -> work_accident.under_review ("check the insurance type" -> "check accident site")
    - work_accident.under_review -> self_employed_person.reported ("check accident site" -> "report employer ")
    - work_accident.under_review -> insured_employment.protected ("check accident site" -> "allocate employer")
    - work_accident.reported -> doctor_visit.scheduled ("Report accident insurance" -> "merge documents")
    - work_accident.under_review -> private_insurance.reported ("verify the person involved" -> "check the insurance type")

### X_g01/3.bpmn2.xml
- cible   : `course.registered`  <- activite matchee : "register for the course" (score=0.7214)
- terme   : `twitter_account.connected`  <- activite matchee : "connect to the twitter" (score=0.7104)
- operateur : AND  |  citation : "The application is also connected to your twitter account, and lets you tweet to friends who might want to join you, you can complete the registration for the course and provide the payment information."
- aretes process touchant ces etats (10) :
    - course.registered -> twitter_account.connected ("register for the course" -> "tweet for your friends")
    - course.registered -> payment_information.provided ("register for the course" -> "receive the customer data")
    - twitter_account.checked -> course.registered ("check the account" -> "register for the course")
    - course.registered -> twitter_account.connected ("register for the course" -> "tweet for your friends")
    - course.registered -> payment_information.provided ("register for the course" -> "receive the customer data")
    - activation_request.responded_to -> course.registered ("wait for the response" -> "register for the course")
    - twitter_account.connected -> payment_information.provided ("connect to the twitter" -> "send the payment infos")
    - course.registered -> payment_information.provided ("register for the course" -> "receive the customer data")
    - twitter_account.registered -> course.registered ("register your acoount" -> "register for the course")
    - payment_information.provided -> twitter_account.connected ("receive the customer data" -> "connect to the twitter")


## Modele : gpt-oss-20b

### G_g03/3.bpmn2.xml
- cible   : `bank_account.entered`  <- activite matchee : "enter account information" (score=0.6526)
- terme   : `payment_method.selected`  <- activite matchee : "select payment method" (score=0.865)
- operateur : AND  |  citation : "If you choose your bank account, enter your IBAN and BIC numbers."
- aretes process touchant ces etats (5) :
    - confirmation_mail.received -> payment_method.selected ("click link in the confirmation mail" -> "select payment method")
    - account.created -> bank_account.entered ("create account" -> "enter account information")
    - payment_method.selected -> credit_card.entered ("select payment method" -> "enter credit card information")
    - payment_method.selected -> iban.entered ("select payment method" -> "enter IBAN and BIC")
    - bank_account.entered -> confirmation_mail.received ("enter account information" -> "receive confirmation mail")

### G_g03/3.bpmn2.xml
- cible   : `race.selected`  <- activite matchee : "select race of character" (score=0.6813)
- terme   : `realm.selected`  <- activite matchee : "select realm of character" (score=0.7885)
- operateur : AND  |  citation : "After that you can log into the game and select realm, race and class of your character."
- aretes process touchant ces etats (6) :
    - realm.selected -> character_name.entered ("select realm of character" -> "enter name")
    - race.selected -> character_name.entered ("select race of character" -> "enter name")
    - race.selected -> character_name.entered ("select class of character" -> "enter name")
    - game.logged_in -> race.selected ("log into game" -> "select race of character")
    - game.logged_in -> realm.selected ("log into game" -> "select realm of character")
    - game.logged_in -> race.selected ("log into game" -> "select class of character")

### G_g03/4.bpmn2.xml
- cible   : `battle_net_account.created`  <- activite matchee : "Check battle.net account" (score=0.6922)
- terme   : `account.created`  <- activite matchee : "Click link in mail to create account" (score=0.4978)
- operateur : AND  |  citation : "you enter the account information"
- aretes process touchant ces etats (7) :
    - battle_net_account.created -> character_name.entered ("Create new WoW character" -> "Enter character name")
    - battle_net_account.created -> battle_net_account.created ("Create new WoW character" -> "Check battle.net account")
    - battle_net_account.created -> bank_account.entered ("Check battle.net account" -> "Enter account information")
    - battle_net_account.created -> wow_subscription.active ("Check battle.net account" -> "Check WoW subscription")
    - account.created -> wow_subscription.active ("Click link in mail to create account" -> "Check WoW subscription")
    - bank_account.entered -> account.created ("Enter account information" -> "Click link in mail to create account")
    - bank_account.entered -> account.created ("Enter account information" -> "Click link in mail to create account")

### G_g03/4.bpmn2.xml
- cible   : `bank_account.entered`  <- activite matchee : "Enter account information" (score=0.6631)
- terme   : `payment_method.selected`  <- activite matchee : "Select payment method" (score=0.8836)
- operateur : AND  |  citation : "If you choose your bank account, enter your IBAN and BIC numbers."
- aretes process touchant ces etats (6) :
    - wow_subscription.active -> payment_method.selected ("Check WoW subscription" -> "Select payment method")
    - battle_net_account.created -> bank_account.entered ("Check battle.net account" -> "Enter account information")
    - payment_method.selected -> iban.entered ("Select payment method" -> "Enter IBAN and BIC numbers")
    - payment_method.selected -> credit_card.entered ("Select payment method" -> "Enter credit card information")
    - bank_account.entered -> account.created ("Enter account information" -> "Click link in mail to create account")
    - bank_account.entered -> account.created ("Enter account information" -> "Click link in mail to create account")

### G_g03/4.bpmn2.xml
- cible   : `character_name.entered`  <- activite matchee : "Enter character name" (score=0.8011)
- terme   : `account.created`  <- activite matchee : "Click link in mail to create account" (score=0.4978)
- operateur : AND  |  citation : "While you are setting up your account, you can already come up with good character names."
- aretes process touchant ces etats (6) :
    - battle_net_account.created -> character_name.entered ("Create new WoW character" -> "Enter character name")
    - character_name.entered -> character_name.entered ("Enter character name" -> "Enter character name")
    - character_name.entered -> confirmation.confirmed ("Enter character name" -> "Confirmation of name")
    - account.created -> wow_subscription.active ("Click link in mail to create account" -> "Check WoW subscription")
    - bank_account.entered -> account.created ("Enter account information" -> "Click link in mail to create account")
    - bank_account.entered -> account.created ("Enter account information" -> "Click link in mail to create account")

### G_g03/5.bpmn2.xml
- cible   : `character_name.available`  <- activite matchee : "Coming up with the name of the Charachter" (score=0.5778)
- terme   : `character_name.entered`  <- activite matchee : "Enter name for the Character" (score=0.807)
- operateur : AND  |  citation : "until a name is still available."
- aretes process touchant ces etats (3) :
    - character_name.available -> character_name.entered ("Coming up with the name of the Charachter" -> "Enter name for the Character")
    - character_name.entered -> confirmation.received ("Enter name for the Character" -> "Get confirmation and selfies of your character")
    - realm.selected -> character_name.entered ("Select a Realm, Race, Class of your Charachter" -> "Enter name for the Character")

### G_g03/8.bpmn2.xml
- cible   : `payment_method.selected`  <- activite matchee : "Validate payment option" (score=0.6176)
- terme   : `wow_subscription.active`  <- activite matchee : "WoW system" (score=0.4188)
- operateur : AND  |  citation : "If not, you can select the payment method."
- aretes process touchant ces etats (6) :
    - account.entered -> payment_method.selected ("Create new account" -> "Provide payment data")
    - character_name.entered -> wow_subscription.active ("Type in name of character" -> "WoW system")
    - payment_method.selected -> confirmation.received ("Send payment accepted message" -> "feedback")
    - payment_method.selected -> payment_method.selected ("Provide payment data" -> "Payment system")
    - payment_method.selected -> confirmation.received ("Provide payment data" -> "feedback")
    - payment_method.selected -> payment_method.selected ("Validate payment option" -> "Send payment accepted message")

### G_g03/8.bpmn2.xml
- cible   : `game.logged_in`  <- activite matchee : "Log in to WoW" (score=0.452)
- terme   : `wow_subscription.active`  <- activite matchee : "WoW system" (score=0.4188)
- operateur : AND  |  citation : "After that you can log into the game"
- aretes process touchant ces etats (4) :
    - confirmation.received -> game.logged_in ("feedback" -> "Log in to WoW")
    - character_name.entered -> wow_subscription.active ("Type in name of character" -> "WoW system")
    - game.logged_in -> character_name.available ("Log in to WoW" -> "Specify character information")
    - battle_net_account.checked -> game.logged_in ("Check WoW account" -> "Log in to WoW")

### M_g01/0.bpmn2.xml
- cible   : `sketches.sent`  <- activite matchee : "Send sketches and further information" (score=0.7235)
- terme   : `project.created`  <- activite matchee : "Create Project" (score=0.7388)
- operateur : AND  |  citation : "First you have to send him several sketches, and then tell him what to change until you are satisfied with the result."
- aretes process touchant ces etats (4) :
    - sketches.sent -> stl.sent ("Send sketches and further information" -> "Receive STL File")
    - sketches.sent -> color.chosen ("Send sketches and further information" -> "Choose plastic color")
    - sketches.sent -> project.created ("Send sketches and further information" -> "Create Project")
    - project.created -> gcode_file.generated ("Create Project" -> "Generate STL")

### M_g01/10.bpmn2.xml
- cible   : `color.checked`  <- activite matchee : "Check  how much color is left" (score=0.4648)
- terme   : `color.in_stock`  <- activite matchee : "Enough color available" (score=0.5952)
- operateur : AND  |  citation : "If you have the color at home (in stock), you check how much color you have left."
- aretes process touchant ces etats (7) :
    - color.on_shopping_list -> color.in_stock ("Put color on shopping list" -> "Enough color available")
    - color.in_stock -> stl.sent ("Enough color available" -> "Get STL File")
    - color.in_stock -> extruder.heated ("Enough color available" -> "Turn on the printer")
    - color.checked -> color.on_shopping_list ("Check  how much color is left" -> "Put color on shopping list")
    - color.checked -> color.in_stock ("Check  how much color is left" -> "Enough color available")
    - color.chosen -> color.checked ("Choose plastic color" -> "Check  how much color is left")
    - color.ordered -> color.in_stock ("Order color" -> "Enough color available")

### R_j02/3.bpmn2.xml
- cible   : `questions.asked`  <- activite matchee : "Ask questions" (score=0.8268)
- terme   : `inspection.begun`  <- activite matchee : "Inspection done" (score=0.8254)
- operateur : AND  |  citation : "Questions are asked, and you have to input values."
- aretes process touchant ces etats (12) :
    - machine.entered -> questions.asked ("Enter type of machine and serial number" -> "Answer questions")
    - questions.asked -> machine.entered ("Ask questions" -> "Autofill values relevant to machine")
    - questions.asked -> questions.asked ("Ask questions" -> "Answer questions")
    - questions.asked -> questions.asked ("Question 1" -> "Ask questions")
    - questions.asked -> questions.asked ("Question 3" -> "Ask questions")
    - values.collected -> questions.asked ("Receive type and serial number" -> "Query relevant questions")
    - questions.asked -> buttons.pressed ("Answer questions" -> "Hit next")
    - questions.asked -> questions.asked ("Query relevant questions" -> "Question 2")
    - questions.asked -> questions.asked ("Query relevant questions" -> "Question 3")
    - questions.asked -> questions.asked ("Query relevant questions" -> "Question 1")
    - values.shown -> inspection.begun ("Confirm values" -> "Finish inspection")
    - questions.asked -> questions.asked ("Question 2" -> "Ask questions")

### R_j02/4.bpmn2.xml
- cible   : `questions.asked`  <- activite matchee : "Ask Question" (score=0.7149)
- terme   : `inspection.begun`  <- activite matchee : "send in inspection results" (score=0.4823)
- operateur : AND  |  citation : "Questions are asked, and you have to input values."
- aretes process touchant ces etats (7) :
    - machine.entered -> questions.asked ("Enter serial number" -> "Ask Question")
    - values.shown -> questions.asked ("input values" -> "Ask Question")
    - machine.entered -> questions.asked ("Enter type of machine" -> "Ask Question")
    - values.collected -> inspection.begun ("evaluate results" -> "send in inspection results")
    - machine.entered -> inspection.begun ("Readjust botteling machine" -> "send in inspection results")
    - questions.asked -> values.collected ("Ask Question" -> "collect values")
    - questions.asked -> values.shown ("Ask Question" -> "input values")

### V_k09/0.bpmn2.xml
- cible   : `stock_management_system.registered`  <- activite matchee : "Connect to the managment system" (score=0.3589)
- terme   : `product.registered`  <- activite matchee : "Register the product in the management system" (score=0.613)
- operateur : AND  |  citation : "After the ordered product has arrived it's registered in the stock management system."
- aretes process touchant ces etats (6) :
    - stock_management_system.registered -> order.read ("Connect to the managment system" -> "Read order from system")
    - product.registered -> order.read ("Register the product in the management system" -> "Read order from system")
    - product.registered -> product.reordered ("Register the product in the management system" -> "Pack the product")
    - product.registered -> shipment.shipped ("Register the product in the management system" -> "Request a courier")
    - delivery_delay_penalty.demanded -> product.registered ("delivery delay penalty from the wholesaler" -> "Register the product in the management system")
    - order.ready_for_shipment -> product.registered ("Give the Ok for the arrived product" -> "Register the product in the management system")

### X_g01/3.bpmn2.xml
- cible   : `twitter_account.registered`  <- activite matchee : "register your acoount" (score=0.5496)
- terme   : `twitter_account.activated`  <- activite matchee : "request an activation" (score=0.4447)
- operateur : AND  |  citation : "request an activation and wait for a response. As soon as you have an account, log into it."
- aretes process touchant ces etats (5) :
    - twitter_account.logged_in -> twitter_account.registered ("check the account" -> "check the uni account")
    - twitter_account.activated -> payment_information.provided ("request an activation" -> "wait for the response")
    - twitter_account.registered -> twitter_account.activated ("check the uni account" -> "request an activation")
    - twitter_account.registered -> twitter_account.registered ("check the uni account" -> "register your acoount")
    - twitter_account.registered -> course_ticket.received ("register your acoount" -> "register for the course")

