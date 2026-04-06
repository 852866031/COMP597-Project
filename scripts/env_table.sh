
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "'${BASH_SOURCE[0]}' is a config file, it should not be executed. Source it to populate the variables."
    exit 1
fi

UPPER_LEFT_CORNER=$(printf "\u250c")
UPPER_RIGHT_CORNER=$(printf "\u2510")
LOWER_LEFT_CORNER=$(printf "\u2514")
LOWER_RIGHT_CORNER=$(printf "\u2518")

HORIZONTAL_BAR=$(printf "\u2500")
VERTICAL_BAR=$(printf "\u2502")

CROSS=$(printf "\u253c")

LEFT_CROSS=$(printf "\u251c")
RIGHT_CROSS=$(printf "\u2524")
TOP_CROSS=$(printf "\u252c")
BOTTOM_CROSS=$(printf "\u2534")

function make_row_delim() {
        local first=$1
        local middle=$2
        local last=$3
        local col_delim=$4
        local num_column=$5
        local column_widths=("$@")
        unset column_widths[0]
        unset column_widths[1]
        unset column_widths[2]
        unset column_widths[3]
        unset column_widths[4]

        local first_col=0
        local row="${first}"
        for w in "${column_widths[@]}"; do
                if [[ "${first_col}" -ne 0 ]]; then
                        row="${row}${col_delim}"
                fi
                local col=$(printf "%${w}s" | sed "s/ /${middle}/g")
                row="${row}${col}"
                first_col=1
        done
        row="${row}${last}"
        echo ${row}
}

UPPER_LEFT_CORNER=$(printf "\u250c")
UPPER_RIGHT_CORNER=$(printf "\u2510")
LOWER_LEFT_CORNER=$(printf "\u2514")
LOWER_RIGHT_CORNER=$(printf "\u2518")

HORIZONTAL_BAR=$(printf "\u2500")
VERTICAL_BAR=$(printf "\u2502")

CROSS=$(printf "\u253c")

LEFT_CROSS=$(printf "\u251c")
RIGHT_CROSS=$(printf "\u2524")
TOP_CROSS=$(printf "\u252c")
BOTTOM_CROSS=$(printf "\u2534")

function make_row_delim() {
        local first=$1
        local middle=$2
        local last=$3
        local col_delim=$4
        local num_column=$5
        local column_widths=("$@")
        unset column_widths[0]
        unset column_widths[1]
        unset column_widths[2]
        unset column_widths[3]
        unset column_widths[4]

        local first_col=0
        local row="${first}"
        for w in "${column_widths[@]}"; do
                if [[ "${first_col}" -ne 0 ]]; then
                        row="${row}${col_delim}"
                fi
                local col=$(printf "%${w}s" | sed "s/ /${middle}/g")
                row="${row}${col}"
                first_col=1
        done
        row="${row}${last}"
        echo ${row}
}

function max_widths() {
        local pattern=$1
        local max_var=0
        local max_val=0
        shopt -s lastpipe
        env | grep -P "${pattern}" | tr "\n" "\0" | while IFS='' read -d '' line ; do
                local var_name=${line%%=*}
                local length=${#var_name}
                if [[ "${length}" -gt "${max_var}" ]]; then
                        max_var=${length}
                fi
                local var_value=${line#*=}
                length=${#var_value}
                if [[ "${length}" -gt "${max_val}" ]]; then
                        max_val=${length}
                fi
        done
        shopt -u lastpipe
        printf "${max_var},${max_val}"
}

function sum_widths() {
        local column_widths=("$@")
        local first_col=0
        local sum=0
        for w in "${column_widths[@]}"; do
                if [[ "${first_col}" -ne 0 ]]; then
                        sum=$((${sum}+1))
                fi
                sum=$((${sum}+${w}))
                first_col=1
        done
        printf ${sum}
}

function table_title() {
        local width=$1
        local title=$2

        local top_title_row=$(make_row_delim ${UPPER_LEFT_CORNER} ${HORIZONTAL_BAR} ${UPPER_RIGHT_CORNER} ${TOP_CROSS} 1 "${width}")
        local padding_length=$((${width}-${#title}))
        local left_padding=$((${padding_length}/2))
        local right_padding=${left_padding}
        if [[ $((${left_padding}+${right_padding})) -ne ${padding_length} ]]; then
                left_padding=$((${left_padding}+1))
        fi
        left_padding=$((${left_padding}+${#title}))
        echo ${top_title_row}
        printf "${VERTICAL_BAR}%${left_padding}s%${right_padding}s${VERTICAL_BAR}\n" "${title}" ""
}

function env_table() {
        local pattern=$1
        local title=$2
        local var_col_name="Variable"
        local val_col_name="Value"
        local tmp=$(max_widths ${pattern})
        local var_name_width=$(echo $tmp | cut -d , -f 1 )
        if [[ ${#var_col_name} -gt $var_name_width ]]; then
                var_name_width=${#var_col_name}
        fi
        local var_val_width=$(echo $tmp | cut -d , -f 2 )
        if [[ ${#val_col_name} -gt ${var_val_width} ]]; then
                var_val_width=${#val_col_name}
        fi
        local col_widths=($((${var_name_width}+2)) $((${var_val_width}+2)))
        local top_row=$(make_row_delim ${LEFT_CROSS} ${HORIZONTAL_BAR} ${RIGHT_CROSS} ${TOP_CROSS} 2 "${col_widths[@]}")
        local mid_row=$(make_row_delim ${LEFT_CROSS} ${HORIZONTAL_BAR} ${RIGHT_CROSS} ${CROSS} 2 "${col_widths[@]}")
        local bot_row=$(make_row_delim ${LOWER_LEFT_CORNER} ${HORIZONTAL_BAR} ${LOWER_RIGHT_CORNER} ${BOTTOM_CROSS} 2 "${col_widths[@]}")
        local row_format="${VERTICAL_BAR} %-${var_name_width}s ${VERTICAL_BAR} %-${var_val_width}s ${VERTICAL_BAR}\n"
        local title_width=$(sum_widths "${col_widths[@]}")

        table_title "${title_width}" "${title}"
        echo "${top_row}"
        printf "${row_format}" "Variable" "Value"
        echo "${mid_row}"
        env | grep -P "${pattern}" | LC_COLLATE=C sort | tr "\n" "\0" | while IFS='' read -d '' line ; do
                local var_name=${line%%=*}
                local var_value=${line#*=}
                printf "${row_format}" ${var_name} ${var_value}
        done
        echo "${bot_row}"
}
