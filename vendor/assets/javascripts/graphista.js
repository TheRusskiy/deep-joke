$(document).ready(function() {
	/*============================================
	Header
	==============================================*/

	$('#home').height($(window).height());
	$('.home-half').height($(window).height() / 2);

	if($('[canvas=container]').scrollTop() > ($(window).height()+50)){
		$('.backstretch').hide();
	}else{
		$('.backstretch').show();
	}

	$('[canvas=container]').scroll( function() {
		var st = $(this).scrollTop(),
			wh = $(window).height(),
			sf = 1.2 - st/(10*wh);

		$('.backstretch img').css({
			'transform' : 'scale('+sf+')',
			'-webkit-transform' : 'scale('+sf+')'
		});

		$('#home .container').css({ 'opacity' : (1.4 - st/400) });

		if($('[canvas=container]').scrollTop() > ($(window).height()+50)){
			$('.backstretch').hide();
		}else{
			$('.backstretch').show();
		}

	});

	var st = $(this).scrollTop(),
		wh = $(window).height(),
		sf = 1.2 - st/(10*wh);

	$('.backstretch img').css({
		'transform' : 'scale('+sf+')',
		'-webkit-transform' : 'scale('+sf+')'
	});

	$('#home .container').css({ 'opacity' : (1.4 - st/400) });


	/*============================================
	Navigation Functions
	==============================================*/
	if ($('[canvas=container]').scrollTop()< ($(window).height()-50)){
		$('#main-nav').removeClass('scrolled');
	}
	else{
		$('#main-nav').addClass('scrolled');
	}

	$('[canvas=container]').scroll(function(){
		if ($('[canvas=container]').scrollTop()< ($(window).height()-50)){
			$('#main-nav').removeClass('scrolled');
		}
		else{
			// console.log("now");
			$('#main-nav').addClass('scrolled');
		}
	});

	/*============================================
	Waypoints Animations
	==============================================*/

	$('.scrollimation').waypoint(function(){
		$(this).addClass('in');
	},{
		offset:'90%',
		context: document.getElementById('main-canvas')
	});

	/*============================================
	Project thumbs - Masonry
	==============================================*/
	$(window).load(function(){

		$('#projects-container').css({visibility:'visible'});

		$('#projects-container').masonry({
			itemSelector: '.project-item:not(.filtered)',
			//columnWidth:370,
			isFitWidth: true,
			isResizable: true,
			isAnimated: !Modernizr.csstransitions,
			gutterWidth: 25
		});
	});
	
	/*============================================
	Skills
	==============================================*/
	$(window).load(function(){
		$('.skills-item').each(function(){
			var perc = $(this).find('.percent').data('percent');
			$(this).data('height',perc);
		})

		$('.touch .skills-item').each(function(){
			$(this).css({'height':$(this).data('height')+'%'});
		})

		$('.touch .skills-bars').css({'opacity':1});
	});
});
