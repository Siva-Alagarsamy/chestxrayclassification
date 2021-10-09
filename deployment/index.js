
function analyze_sentiment() {
    $.ajax({
        url: '/check_sentiment',
        dataType: 'text',
        type: 'post',
        contentType: 'application/json',
        data: JSON.stringify( { "text": $('textarea#mytextarea').val() } ),
        success: function( data, textStatus, jQxhr ){
			response = JSON.parse(data)
            $('#result').text(  "Sentiment score = " + response.result);
			$('#verynegative').hide();
			$('#negative').hide();
			$('#neutral').hide();
			$('#positive').hide();
			$('#verypositive').hide();
			if (response.result <= 0.3) {
				$('#verynegative').show();
			}
			else if(response.result <= 0.48) {
				$('#negative').show();
			}
			else if(response.result <= 0.52) {
				$('#neutral').show();
			}
			else if(response.result <= 0.6) {
				$('#positive').show();
			}
			else {
				$('#verypositive').show();
			}
				
        },
        error: function( jqXhr, textStatus, errorThrown ){
            console.log( errorThrown );
        }
        });
 

}