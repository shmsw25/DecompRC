(function() {

  var paragraph_id = 0;
	var contextss;
  var context_questions;
  var all_contextss;
  var all_questions;
  var titles;
  var mode=0;
  var reasoningType=2;
  var custom_message = ">>> Write My Own Multi-hop Question <<<";

  var chosen_paragraphs=[];

  $( window ).init(function(){
		load();
	});

	function load(){
		sendAjax("/select", {}, (data) => {
			contextss = data.contextss;
	    context_questions = data.context_questions;
      all_contextss = data.all_contextss;
      all_questions = data.all_questions;
      titles = data.titles;
			var dropdown = document.getElementById("question");
			for(var i=0; i<context_questions.length; i++){
				var opt = document.createElement("option");
				opt.value = parseInt(i);
        opt.id = "question-option-"+parseInt(i);
				opt.innerHTML = context_questions[i];
        dropdown.appendChild(opt);
      }

			paragraph_id = 0;
			loadExample();
      dropdown.onchange = function () {
        $("#answer").html('');
				paragraph_id = this.value;
				loadExample();
      };

      $('.editOption').keyup(function () {
        var editText = $('.editOption').val();
        $('editable').val(editText);
        $('editable').html(editText);
      });

      $('.editOption').on('click', function () {
        $('#answer').html('')
      });

      $(".run").click(loadAnswer);

      $('.mode').click(function() {
        mode = parseInt($('.mode:checked').val());
        $('#answer').html('');
        $('#paragraph').html('');
        $('#paragraph').val('');
        if (mode === 0) {
          $('#select-question').show();
          $('#write-question').hide();
          $('.paragraph-choice-container').hide();
          clearParagraphChoices();
          loadExample();
        } else {
          $('#select-question').hide();
          $('#write-question').show();
          $('.paragraph-choice-container').show();
        }
      });

      $('.reasoning-type').click(function() {
        reasoningType = parseInt($('.reasoning-type:checked').val());
        $('#answer').html('');
      });

      $('.paragraph-choice').click(function(){
        $('#answer').html('');
        clearParagraphChoices();
        $('.editOption').val('');
        $('.editOption').html('');

        var chosens = $('.paragraph-choice:checked');
        var container = $('#paragraph-chosen');
        container.html('<div class="label-container"><span class="label label-default">Recommended Paragraphs:</span><div/>');

        for (var i=0; i<chosens.length; i++) {
          var chosen = parseInt(chosens[i].value);
          var paragraphs = all_contextss[chosen];
          for (var j=0; j<paragraphs.length; j++) {
            var val =  chosen.toString() + " " + j.toString();
            container.append('<div class="pretty p-icon p-curve p-rotate">'
                            + '<input type="checkbox" name="radio66" class="paragraph-individual-choice" value="' + val + '">'
                            + '<div class="state p-' + ['danger','warning','success'][chosen] + '-o">'
                            + '<i class="icon glyphicon glyphicon-remove"></i>'
                            + '<label>' + paragraphs[j].title + '</label>'
                            + '</div></div>');
          }
        }
        $('.paragraph-individual-choice').click(loadParagraphChoices);
      });

      /* Label Tooltip */
      $('.mode').mouseover(function(event){
        $('#mode-tooltip').removeClass('tooltip-hidden').addClass('tooltip-visible');
        if (parseInt(event.target.value)===0)
          $('#mode-tooltip').html("You can see example questions & paragraphs from HotpotQA.");
        else
          $('#mode-tooltip').html("You can write your own questions & paragraphs.");
      });
      $('.mode').mouseout(function(){
        $('#mode-tooltip').removeClass('tooltip-visible').addClass('tooltip-hidden');

      });
      $('.reasoning-type').mouseover(function(event){
        $('#reasoning-type-tooltip').removeClass('tooltip-hidden').addClass('tooltip-visible');
        if (parseInt(event.target.value)===2)
          $('#reasoning-type-tooltip').html("You can ask the model to decide on the most suitable reasoning type for you (slower).");
        else
          $('#reasoning-type-tooltip').html("You can specify if the question is bridging or intersection (faster).");
      });
      $('.reasoning-type').mouseout(function(){
        $('#reasoning-type-tooltip').removeClass('tooltip-visible').addClass('tooltip-hidden');
      });
      $('.paragraph-choice').mouseover(function(event){
        $('#paragraph-choice-tooltip').removeClass('tooltip-hidden').addClass('tooltip-visible');
        $('#paragraph-choice-tooltip').html(
          "You can select a category of your interest, and see recommended paragraphs & questions.");
      });
      $('.paragraph-choice').mouseout(function(){
        $('#paragraph-choice-tooltip').removeClass('tooltip-visible').addClass('tooltip-hidden');
      });

      /* Description Button */
      $('#description-button').click(function(){
        if ($('#description-button').html() === 'Show Me Details!') {
          $('#description').show();
          $('#description-button').html('Hide Details!');
        } else {
          $('#description').hide();
          $('#description-button').html('Show Me Details!');
        }
      })
      $('#type-description-button').click(function(){
        if ($('#type-description-button').html() === 'What are they?') {
          $('#type-description').show();
          $('#type-description-button').html('Got it!');
        } else {
          $('#type-description').hide();
          $('#type-description-button').html('What are they?');
        }
      })

		});
  }

  function loadExample(){
    $('#paragraph').html(contextss[paragraph_id].join('\n\n'));
    $('#paragraph').val(contextss[paragraph_id].join('\n\n'));
  }

  function clearParagraphChoices(){
    $('#paragraph').html('');
    $('#paragraph').val('');
    chosen_paragraphs = [];
  }

  function loadParagraphChoices(){
    $('#answer').html();
    var chosens = [[], [], []];
    var recommended_questions = [];
    var checked = $('.paragraph-individual-choice:checked');
    var paragraphs = [];
    $('#paragraph').val().split('\n').forEach(function(d) {
      if (d.length>0)
        paragraphs.push(d);
    });

    // handle paragraph text
    if (this.checked) {
      var indices = this.value.split(' ');
      $('#paragraph').val($('#paragraph').val() + all_contextss[parseInt(indices[0])][parseInt(indices[1])].paragraph + "\n\n");
      chosen_paragraphs.push(this.value);
    } else {
      console.assert(chosen_paragraphs.includes(this.value));
      var index = chosen_paragraphs.indexOf(this.value);
      //console.assert(chosen_paragraphs.length===paragraphs.length);
      chosen_paragraphs.splice(index, 1);
      paragraphs.splice(index, 1);
      $('#paragraph').val(paragraphs.join('\n\n') + '\n\n');
    }

    // find question recommendations
    for (var i=0; i<checked.length; i++) {
      var indices = checked[i].value.split(' ');
      chosens[parseInt(indices[0])].push(parseInt(indices[1]));
      //paragraph_text += all_contextss[parseInt(indices[0])][parseInt(indices[1])].paragraph + "\n\n";
    }
    for (var i=0; i<3; i++) {
      for (var j=0; j<all_questions[i].length; j++) {
        if (chosens[i].indexOf(2*j)>-1 && chosens[i].indexOf(2*j+1)>-1) {
          recommended_questions.push([i, all_questions[i][j]]);
        }
      }
    }

    // display question recommendations
    $('#question-recommendation').html(
      '<div class="label-container"><span class="label label-default">Recommended Multi-hop Questions:</span><div/>');
    if (recommended_questions.length===0)
      $('#question-recommendation').hide();
    else  {
      $('#question-recommendation').show();
      recommended_questions.forEach(function(d){
        $('#question-recommendation').append('<div class="pretty p-icon p-curve p-rotate">'
                            + '<input type="radio" name="radio66" class="question-individual-choice" value="' + d[1] + '">'
                            + '<div class="state p-' + ['danger','warning','success'][d[0]] + '-o">'
                            + '<i class="icon glyphicon glyphicon-remove"></i>'
                            + '<label>' + d[1] + '</label>'
                            + '</div></div>');
      });
      $('.question-individual-choice').click(function(){
        $('.editOption').val($('.question-individual-choice:checked').val());
        $('.editOption').html($('.question-individual-choice:checked').val());
      });
    }
  }

  function loadAnswer(){
    var question_text = $('select#question option:selected').html();
    var paragraphs_text = $('#paragraph').val();
    var reasoningTypeNum = reasoningType;
    if (mode === 1) {
      question_text = $('.editOption').val();
      if (!(question_text.replace(/\s/g, '').length)) {
        alert('Please enter a non-empty question.');
        return;
      }
      if (!(paragraphs_text.replace(/\s/g, '').length)) {
        alert('Please enter a non-empty paragraph.');
        return;
      }
    }
		document.getElementById("answer").innerHTML = "";
		document.getElementById("loading").style.display = "block";
		var data = {
      'paragraphs': paragraphs_text,
      'question': question_text,
      'reasoningType': reasoningTypeNum
    };
		sendAjax("/submit", data, (answer) => {
			document.getElementById("loading").style.display = "none";
			var answer_field = document.getElementById('answer');
			answer_field.appendChild(getPanel("Question Type", answer.q_type));
			answer_field.appendChild(getPanel("SubQuestion1", answer.subq1));
			if (answer.q_type === "Bridging")
				answer_field.appendChild(getPanel("Answer to SubQuestion1", answer.answer1));
			answer_field.appendChild(getPanel("SubQuestion2", answer.subq2));
			answer_field.appendChild(getPanel("Final Answer", answer.answer2));
		});
	}
  function sendAjax(url, data, handle){
		$.getJSON(url, data, function(response){
			handle(response.result);
		});
	}

	function getPanel(heading_text, context_text){
		var div = document.createElement('div');
		div.className = "panel panel-default";
		var heading = document.createElement('div');
		heading.className = "panel-heading";
		heading.innerHTML = heading_text;
		var context = document.createElement('div');
		context.className = "panel-body";
		context.innerHTML = context_text;
		div.appendChild(heading);
		div.appendChild(context);
		return div
	}

})();



